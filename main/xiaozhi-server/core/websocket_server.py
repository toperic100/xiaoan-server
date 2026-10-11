import asyncio
import logging
from urllib.parse import parse_qs, urlparse

import websockets
from config.logger import setup_logging


class SuppressInvalidHandshakeFilter(logging.Filter):
    """过滤明显的无效 HTTP/HTTPS 探测，避免污染 WebSocket 日志。"""

    def filter(self, record):
        msg = record.getMessage()
        suppress_keywords = [
            "opening handshake failed",
            "did not receive a valid HTTP request",
            "connection closed while reading HTTP request",
            "line without CRLF",
        ]
        return not any(keyword in msg for keyword in suppress_keywords)


def _setup_websockets_logger():
    filter_instance = SuppressInvalidHandshakeFilter()
    for logger_name in (
        "websockets",
        "websockets.server",
        "websockets.client",
    ):
        logging.getLogger(logger_name).addFilter(filter_instance)


_setup_websockets_logger()

from core.connection import ConnectionHandler
from config.config_loader import get_config_from_api_async
from core.auth import AuthManager, AuthenticationError
from core.utils.modules_initialize import initialize_modules
from core.utils.util import check_vad_update, check_asr_update

TAG = __name__


class WebSocketServer:
    """
    WebSocket Server V4

    目标：
    1. WebSocket 监听必须先建立，AI 模块初始化不得阻塞端口启动。
    2. VAD / ASR / LLM / Memory / Intent 的同步初始化全部放入线程池。
    3. WebSocket 握手、设备参数解析、认证与 AI 模块加载解耦。
    4. 保留 ConnectionHandler / AFE / 音频协议，不在本层改动音频链路。
    5. WebSocket 连接异常不得拖垮服务器主事件循环。
    6. 配置热更新采用“先构建、后一次性切换”，失败时保留旧状态。
    """

    MODULE_NOT_READY_CLOSE_CODE = 1013
    INVALID_REQUEST_CLOSE_CODE = 1008
    SERVER_ERROR_CLOSE_CODE = 1011

    def __init__(self, config: dict):
        # ---------------------------------------------------------
        # P0：构造函数必须轻量。
        # 这里绝不执行 initialize_modules()。
        # ---------------------------------------------------------
        self.config = config
        self.logger = setup_logging(config)
        self.config_lock = asyncio.Lock()

        self._vad = None
        self._asr = None
        self._llm = None
        self._intent = None
        self._memory = None

        # 模块初始化状态
        self.modules_ready = asyncio.Event()
        self.modules_init_done = asyncio.Event()
        self.module_init_error = None
        self._module_init_task = None
        self._server_started = False

        # 允许通过 server.module_wait_timeout 调整。
        # 默认 15 秒，避免把等待时间硬编码成 8 秒。
        self.module_wait_timeout = self._get_module_wait_timeout(config)

        # 认证状态也在构造阶段只做轻量初始化。
        self._auth_enable, self._allowed_devices, self._auth = (
            self._build_auth_state(config)
        )
        self._sync_auth_attributes()

    @staticmethod
    def _get_module_wait_timeout(config: dict) -> float:
        value = config.get("server", {}).get("module_wait_timeout", 15)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 15.0
        return max(0.0, value)

    def _build_auth_state(self, config: dict):
        auth_config = config.get("server", {}).get("auth", {})
        auth_enable = auth_config.get("enabled", False)
        allowed_devices = set(auth_config.get("allowed_devices", []))
        secret_key = config["server"]["auth_key"]
        expire_seconds = auth_config.get("expire_seconds", None)
        auth = AuthManager(
            secret_key=secret_key,
            expire_seconds=expire_seconds,
        )
        return auth_enable, allowed_devices, auth

    def _sync_auth_attributes(self):
        # 保留旧代码可能直接访问的属性名，降低兼容风险。
        self.auth_enable = self._auth_enable
        self.allowed_devices = self._allowed_devices
        self.auth = self._auth

    def _build_modules_sync(self, config: dict):
        selected_module = config.get("selected_module", [])
        return initialize_modules(
            self.logger,
            config,
            "VAD" in selected_module,
            "ASR" in selected_module,
            "LLM" in selected_module,
            False,
            "Memory" in selected_module,
            "Intent" in selected_module,
        )

    @staticmethod
    def _extract_modules(modules: dict):
        return (
            modules.get("vad"),
            modules.get("asr"),
            modules.get("llm"),
            modules.get("intent"),
            modules.get("memory"),
        )

    def _apply_modules(self, modules: dict):
        (
            self._vad,
            self._asr,
            self._llm,
            self._intent,
            self._memory,
        ) = self._extract_modules(modules)

    def _get_config_snapshot(self):
        # 当前 config 主要是嵌套 dict。这里复制顶层及 server，
        # 避免后台线程直接依赖后续可能被替换的 self.config 对象。
        snapshot = dict(self.config)
        if isinstance(self.config.get("server"), dict):
            snapshot["server"] = dict(self.config["server"])
        if isinstance(self.config.get("selected_module"), list):
            snapshot["selected_module"] = list(self.config["selected_module"])
        return snapshot

    async def initialize_modules_async(self, config_snapshot=None) -> bool:
        """
        在线程池中构建 AI 模块。
        注意：asyncio.to_thread 只把同步工作移出 Event Loop，
        不改变 initialize_modules 本身的同步实现。
        """
        if config_snapshot is None:
            config_snapshot = self._get_config_snapshot()

        self.modules_init_done.clear()
        self.module_init_error = None

        try:
            self.logger.bind(tag=TAG).info(
                "开始后台初始化 VAD / ASR / LLM / Memory / Intent 模块..."
            )

            modules = await asyncio.to_thread(
                self._build_modules_sync,
                config_snapshot,
            )

            self._apply_modules(modules)
            self.module_init_error = None
            self.modules_ready.set()

            self.logger.bind(tag=TAG).info(
                "AI 模块初始化完成，modules_ready=True"
            )
            return True

        except asyncio.CancelledError:
            self.logger.bind(tag=TAG).warning(
                "AI 模块初始化任务被取消"
            )
            raise

        except Exception as e:
            self.modules_ready.clear()
            self.module_init_error = e
            self.logger.bind(tag=TAG).error(
                f"AI 模块初始化失败: {type(e).__name__}: {e}"
            )
            return False

        finally:
            self.modules_init_done.set()

    async def start(self):
        """
        关键启动顺序：

        1. websockets.serve() 先开始监听。
        2. 再启动 AI 模块后台初始化。
        3. 因此 initialize_modules() 再慢，也不会阻塞 WebSocket 端口建立。
        """
        server_config = self.config["server"]
        host = server_config.get("ip", "0.0.0.0")
        port = int(server_config.get("port", 8000))

        self.logger.bind(tag=TAG).info(
            f"WebSocketServer V4 正在启动，监听地址: {host}:{port}"
        )

        async with websockets.serve(
            self._handle_connection,
            host,
            port,
            process_request=self._http_response,
            ping_interval=None,
        ):
            self._server_started = True

            initial_config = self._get_config_snapshot()
            self._module_init_task = asyncio.create_task(
                self.initialize_modules_async(initial_config),
                name="websocket-server-module-init",
            )

            self.logger.bind(tag=TAG).info(
                "WebSocket 端口级监听已就绪，AI 模块正在后台线程加载..."
            )

            try:
                await asyncio.Future()
            finally:
                self._server_started = False

                if self._module_init_task is not None:
                    if not self._module_init_task.done():
                        self._module_init_task.cancel()
                        try:
                            await self._module_init_task
                        except asyncio.CancelledError:
                            pass
                    self._module_init_task = None

    def _get_connection_params(self, websocket):
        """
        device-id 优先从 HTTP Header 获取。
        为兼容旧客户端，也支持 query string。

        注意：
        如果 device-id 来自 query，尽量同步写回 request.headers，
        这样旧版 ConnectionHandler 仍可以继续读取 device-id。
        """
        headers = dict(websocket.request.headers)
        if headers.get("device-id"):
            return headers

        request_path = getattr(websocket.request, "path", None)
        if not request_path:
            return None

        parsed_url = urlparse(request_path)
        query_params = parse_qs(parsed_url.query)

        device_id = query_params.get("device-id", [None])[0]
        if not device_id:
            return None

        extra_headers = {
            "device-id": device_id,
        }

        for key in ("client-id", "authorization"):
            value = query_params.get(key, [None])[0]
            if value is not None:
                extra_headers[key] = value

        # websockets 新版本的 Headers 通常支持 update；
        # 如果底层对象不可变，则不影响本次认证，因为返回值已经包含参数。
        try:
            websocket.request.headers.update(extra_headers)
        except Exception:
            pass

        headers.update(extra_headers)
        return headers

    async def _handle_connection(self, websocket):
        # ---------------------------------------------------------
        # 1. 参数解析
        # ---------------------------------------------------------
        headers = self._get_connection_params(websocket)
        if headers is None:
            self.logger.bind(tag=TAG).warning(
                "拒绝 WebSocket 连接：缺少 device-id"
            )
            await self._safe_close(
                websocket,
                code=self.INVALID_REQUEST_CLOSE_CODE,
                reason="Missing device-id",
            )
            return

        # ---------------------------------------------------------
        # 2. 先认证，再等待 AI。
        #
        # 这样未认证连接不会因为 AI 初始化而长期占用等待槽。
        # ---------------------------------------------------------
        try:
            await self._handle_auth(websocket, headers)
        except AuthenticationError as e:
            self.logger.bind(tag=TAG).warning(
                f"WebSocket 认证失败: {e}"
            )
            await self._safe_close(
                websocket,
                code=self.INVALID_REQUEST_CLOSE_CODE,
                reason="Authentication failed",
            )
            return

        # ---------------------------------------------------------
        # 3. AI 模块尚未就绪时等待。
        #
        # 这里不再“立即 1013”。
        # 但也不能无限等待：如果初始化失败，设备会得到明确结果。
        # ---------------------------------------------------------
        if not self.modules_ready.is_set():
            if self.module_init_error is not None:
                self.logger.bind(tag=TAG).error(
                    "AI 模块初始化已经失败，拒绝建立业务会话"
                )
                await self._safe_close(
                    websocket,
                    code=self.SERVER_ERROR_CLOSE_CODE,
                    reason="Server module initialization failed",
                )
                return

            self.logger.bind(tag=TAG).info(
                f"AI 模块尚未就绪，保持 WebSocket 连接等待，"
                f"timeout={self.module_wait_timeout:.1f}s"
            )

            try:
                await asyncio.wait_for(
                    self.modules_ready.wait(),
                    timeout=self.module_wait_timeout,
                )
            except asyncio.TimeoutError:
                self.logger.bind(tag=TAG).error(
                    "等待 AI 模块初始化超时，关闭当前业务连接"
                )
                await self._safe_close(
                    websocket,
                    code=self.MODULE_NOT_READY_CLOSE_CODE,
                    reason="Server initialize timeout",
                )
                return

            if self.module_init_error is not None:
                await self._safe_close(
                    websocket,
                    code=self.SERVER_ERROR_CLOSE_CODE,
                    reason="Server module initialization failed",
                )
                return

        # ---------------------------------------------------------
        # 4. 创建独立 ConnectionHandler。
        #
        # 不修改 AFE / 音频 / WebSocket 音频协议。
        # ---------------------------------------------------------
        self.logger.bind(tag=TAG).info(
            "客户端认证成功，AI 模块已就绪，创建 ConnectionHandler"
        )

        handler = ConnectionHandler(
            self.config,
            self._vad,
            self._asr,
            self._llm,
            self._memory,
            self._intent,
            self,
        )

        try:
            await handler.handle_connection(websocket)

        except asyncio.CancelledError:
            raise

        except Exception as e:
            self.logger.bind(tag=TAG).error(
                f"处理 WebSocket 会话期间发生未捕获异常: "
                f"{type(e).__name__}: {e}"
            )

        finally:
            await self._safe_close(websocket)

    async def _safe_close(self, websocket, code=None, reason=None):
        try:
            state = getattr(websocket, "state", None)
            if state is not None:
                state_name = getattr(
                    state,
                    "name",
                    str(state),
                ).upper()
                if state_name == "CLOSED":
                    return

            kwargs = {}
            if code is not None:
                kwargs["code"] = code
            if reason is not None:
                kwargs["reason"] = reason

            await websocket.close(**kwargs)

        except Exception as close_error:
            self.logger.bind(tag=TAG).debug(
                f"WebSocket close ignored: "
                f"{type(close_error).__name__}: {close_error}"
            )

    async def _http_response(self, websocket, request_headers):
        """
        兼容：
            Connection: Upgrade
        以及：
            Connection: keep-alive, Upgrade

        同时检查：
            Upgrade: websocket

        非 WebSocket 请求只返回简单 HTTP 200，不进入业务连接逻辑。
        """
        connection_header = request_headers.headers.get(
            "connection",
            "",
        )
        upgrade_header = request_headers.headers.get(
            "upgrade",
            "",
        )

        connection_tokens = {
            token.strip().lower()
            for token in connection_header.split(",")
            if token.strip()
        }

        is_websocket_upgrade = (
            "upgrade" in connection_tokens
            and upgrade_header.strip().lower() == "websocket"
        )

        if is_websocket_upgrade:
            return None

        return websocket.respond(
            200,
            "Server is running\n",
        )

    async def update_config(self) -> bool:
        """
        配置热重载。

        关键原则：
        1. 获取新配置不锁住整个更新流程。
        2. 新 AI 模块在线程池中构建。
        3. 新 AuthManager 提前构建。
        4. 所有准备工作成功后，再一次性切换引用。
        5. 任一步失败，都保留旧配置、旧模块、旧认证状态。
        """
        try:
            async with self.config_lock:
                old_config = self._get_config_snapshot()

            new_config = await get_config_from_api_async(old_config)

            if new_config is None:
                self.logger.bind(tag=TAG).error(
                    "获取新配置失败，保留旧配置"
                )
                return False

            self.logger.bind(tag=TAG).info(
                "获取新配置成功，开始准备热切换"
            )

            update_vad = check_vad_update(
                old_config,
                new_config,
            )
            update_asr = check_asr_update(
                old_config,
                new_config,
            )

            # 在锁外构建，避免长时间阻塞其他协程。
            def build_new_modules():
                selected_module = new_config.get(
                    "selected_module",
                    [],
                )
                return initialize_modules(
                    self.logger,
                    new_config,
                    update_vad,
                    update_asr,
                    "LLM" in selected_module,
                    False,
                    "Memory" in selected_module,
                    "Intent" in selected_module,
                )

            self.logger.bind(tag=TAG).info(
                "开始在线程池构建新的 AI 模块..."
            )

            modules = await asyncio.to_thread(
                build_new_modules
            )

            # AuthManager 也提前准备。
            (
                new_auth_enable,
                new_allowed_devices,
                new_auth,
            ) = self._build_auth_state(new_config)

            new_module_wait_timeout = self._get_module_wait_timeout(
                new_config
            )

            # 所有新状态都准备成功后，才进入极短的切换区。
            async with self.config_lock:
                self._apply_modules(modules)

                self.config = new_config

                self._auth_enable = new_auth_enable
                self._allowed_devices = new_allowed_devices
                self._auth = new_auth
                self.module_wait_timeout = new_module_wait_timeout
                self._sync_auth_attributes()

                self.module_init_error = None
                self.modules_ready.set()

            self.logger.bind(tag=TAG).info(
                "配置、AI 模块与认证状态已完成热切换"
            )
            return True

        except asyncio.CancelledError:
            raise

        except Exception as e:
            self.logger.bind(tag=TAG).error(
                f"更新服务器配置失败: {type(e).__name__}: {e}"
            )
            return False

    async def _handle_auth(
        self,
        websocket,
        headers=None,
    ):
        if not self.auth_enable:
            return

        if headers is None:
            headers = dict(websocket.request.headers)

        device_id = headers.get("device-id")
        client_id = headers.get("client-id")

        if self.allowed_devices and device_id in self.allowed_devices:
            return

        token = headers.get("authorization", "")

        if token.startswith("Bearer "):
            token = token[7:]
        else:
            raise AuthenticationError(
                "Missing or invalid Authorization header"
            )

        auth_success = self.auth.verify_token(
            token,
            client_id=client_id,
            username=device_id,
        )

        if not auth_success:
            raise AuthenticationError("Invalid token")
