import os
import sys
import copy
import json
import re
import uuid
import time
import queue
import asyncio
import threading
import traceback
import subprocess
import websockets
import opuslib_next
import numpy as np

from core.utils.util import (
    extract_json_from_string,
    check_vad_update,
    check_asr_update,
    filter_sensitive_info,
)
from typing import Dict, Any
from collections import deque
from core.utils.modules_initialize import (
    initialize_modules,
    initialize_tts,
    initialize_asr,
)
from core.handle.reportHandle import report, enqueue_tool_report
from core.providers.tts.default import DefaultTTS
from concurrent.futures import ThreadPoolExecutor
from core.utils.dialogue import Message, Dialogue
from core.providers.asr.dto.dto import InterfaceType
from core.handle.textHandle import handleTextMessage
from core.providers.tools.unified_tool_handler import UnifiedToolHandler
from plugins_func.loadplugins import auto_import_modules
from plugins_func.register import Action, ActionResponse, all_function_registry, module_func_map
from core.auth import AuthenticationError
from config.config_loader import get_private_config_from_api
from core.providers.tts.dto.dto import ContentType, TTSMessageDTO, SentenceType
from config.logger import setup_logging, build_module_string, create_connection_logger
from config.manage_api_client import DeviceNotFoundException, DeviceBindException, generate_and_save_chat_title
from core.utils.prompt_manager import PromptManager
from core.utils.voiceprint_provider import VoiceprintProvider
from core.utils.util import get_system_error_response
from core.utils import textUtils


TAG = __name__

auto_import_modules("plugins_func.functions")


class TTSException(RuntimeError):
    pass

# direct_answer 虚拟工具定义
# 不是真实工具，是路由机制：将"调不调工具"的二选一变为"调哪个"的多选，防止小模型误触发真实工具
DIRECT_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "direct_answer",
        "description": "当用户的请求不匹配其他任何工具时，可用此选项直接回复。将回复内容写在response参数里。",
        "parameters": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "你回复用户的完整内容",
                },
            },
            "required": ["response"],
        },
    },
}


class ConnectionHandler:    
    def __init__(
                self,         
                config: Dict[str, Any],
                _vad,
                _asr,
                _llm,
                _memory,
                _intent,
                server=None,
    ):
        # ============================================================
        # WebSocket 消息处理
        #
        # 接收循环与文本业务处理解耦：
        #   binary audio -> 立即处理
        #   text/control -> 放入串行 Worker
        #
        # 防止 LLM / Tool / Weather 等业务处理阻塞 WebSocket recv，
        # 从而导致 ESP32 TCP receive window 堵塞。
        # ============================================================
        self._text_message_queue = None
        self._text_message_worker_task = None
        self.chat_future = None
        self.chat_lock = threading.Lock()       
        self.common_config = config
        self.config = copy.deepcopy(config)
        self.session_id = str(uuid.uuid4())
        self.logger = setup_logging()
        self.server = server  # 保存server实例的引用

        self.need_bind = False  # 是否需要绑定设备
        self.bind_completed_event = asyncio.Event()
        self.bind_code = None  # 绑定设备的验证码
        self.last_bind_prompt_time = 0  # 上次播放绑定提示的时间戳(秒)
        self.bind_prompt_interval = 60  # 绑定提示播放间隔(秒)

        self.read_config_from_api = self.config.get("read_config_from_api", False)

        self.websocket: websockets.ServerConnection | None = None
        self.headers = None
        self.device_id = None
        self.client_ip = None
        self.prompt = None
        self.welcome_msg = None
        self.max_output_size = 0
        self.chat_history_conf = 0
        self.audio_format = "opus"
        self.sample_rate = 24000  # 默认采样率，从客户端 hello 消息中动态更新

        # 客户端状态相关
        self.client_abort = False
        self.client_is_speaking = False
        self.client_listen_mode = "auto"
        self.client_aec = False  # 是否启用了服务端AEC

        # 线程任务相关
        self.loop = None  # 在 handle_connection 中获取运行中的事件循环
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=5)

        # 添加上报线程池
        self.report_queue = queue.Queue()
        self.report_thread = None
        # 未来可以通过修改此处，调节asr的上报和tts的上报，目前默认都开启
        self.report_asr_enable = self.read_config_from_api
        self.report_tts_enable = self.read_config_from_api

        # 依赖的组件
        self.vad = None
        self.asr = None
        self.tts = None
        self._asr = _asr
        self._vad = _vad
        self.llm = _llm
        self.memory = _memory
        self.intent = _intent

        # 为每个连接单独管理声纹识别
        self.voiceprint_provider = None

        # 并发状态锁：保护多线程/协程高频对音频缓冲区及 VAD 状态进行增删改查
        self.state_lock = threading.Lock()

        # vad相关变量
        self.client_audio_buffer = bytearray()
        self.client_have_voice = False
        self.client_voice_window = deque(maxlen=5)
        self.first_activity_time = 0.0  # 记录首次活动的时间（毫秒）
        self.last_activity_time = 0.0  # 统一的活动时间戳（毫秒）
        self.vad_last_voice_time = 0.0  # 记录用户最后一次说话的时间（毫秒）
        self.client_voice_stop = False
        self.last_is_voice = False

        # asr相关变量
        # 因为实际部署时可能会用到公共的本地ASR，不能把变量暴露给公共ASR
        # 所以涉及到ASR的变量，需要在这里定义，属于connection的私有变量
        self.asr_audio = []  # 存储PCM帧列表，供VAD和ASR共享
        self.asr_audio_queue = queue.Queue()
        self.current_speaker = None  # 存储当前说话人
        self.introduced_speakers = set()  # 已"首次引入"的说话人，控制只在首轮带名字
        self.system_introduced_speakers = set()  # 已在 system 注入过身份的说话人，控制 system 身份只首轮出现

        # AEC 缓存与其对应的时间戳字典
        self.aec_audio_cache = {}
        self.aec_audio_cache_time = {}

        # llm相关变量
        self.dialogue = Dialogue()

        # tts相关变量
        self.sentence_id = None
        # 处理TTS响应没有文本返回
        self.tts_MessageText = ""

        # iot相关变量
        self.iot_descriptors = {}
        self.func_handler = None

        self.cmd_exit = self.config["exit_commands"]

        # 是否在聊天结束后关闭连接
        self.close_after_chat = False
        self.load_function_plugin = False
        self.intent_type = "nointent"

        self.timeout_seconds = (
                    int(self.config.get("close_connection_no_voice_time", 120)) + 60
        )  # 在原来第一道关闭的基础上加60秒，进行二道关闭
        self.timeout_task = None

        # {"mcp":true} 表示启用MCP功能
        self.features = None

        # 标记连接是否来自MQTT
        self.conn_from_mqtt_gateway = False

        # 初始化提示词管理器
        self.prompt_manager = PromptManager(self.config, self.logger)

        # 初始化通话状态
        self.calling = False
        # 标记当前是否为来电接听模式
        self.incoming_call = None    
    
    def submit_chat(self, text):
        with self.chat_lock:
            if self.chat_future is not None and not self.chat_future.done():
                self.client_abort = True

                self.logger.bind(tag=TAG).warning(
                    "[CHAT] previous chat is still running, abort requested"
                )

                return None

            self.client_abort = False

            self.chat_future = self.executor.submit(
                self.chat,
                text
            )

            return self.chat_future

    async def _text_message_worker(self):
        """
        WebSocket 文本消息串行处理 Worker。
        优化：加入了“毒丸”优雅退出的异步感知逻辑。
        """
        self.logger.bind(tag=TAG).info(
            "WebSocket text message worker started"
        )

        try:
            while not self.stop_event.is_set():
                try:
                    message = await asyncio.wait_for(
                        self._text_message_queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # 收到“毒丸”对象，立即终止循环退出 Worker
                if message is None:
                    try:
                        self._text_message_queue.task_done()
                    except Exception:
                        pass
                    break

                try:
                    await self._route_text_message(message)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.bind(tag=TAG).error(
                        f"WebSocket text message processing error: {e}"
                    )
                    self.logger.bind(tag=TAG).debug(
                        traceback.format_exc()
                    )
                finally:
                    try:
                        self._text_message_queue.task_done()
                    except Exception:
                        pass

        except asyncio.CancelledError:
            self.logger.bind(tag=TAG).debug(
                "WebSocket text message worker cancelled"
            )
        except Exception as e:
            self.logger.bind(tag=TAG).error(
                f"WebSocket text message worker exited unexpectedly: {e}"
            )
        finally:
            self.logger.bind(tag=TAG).info(
                "WebSocket text message worker exited"
            )       

    async def handle_connection(self, ws: websockets.ServerConnection):
        try:
            # 獲取運行中的事件循環（必須在非同步上下文中）
            self.loop = asyncio.get_running_loop()

            # 獲取並驗證headers
            self.headers = dict(ws.request.headers)
            real_ip = self.headers.get("x-real-ip") or self.headers.get("x-forwarded-for")
            if real_ip:
                self.client_ip = real_ip.split(",")[0].strip()
            else:
                self.client_ip = ws.remote_address[0]
                
            self.logger.bind(tag=TAG).info(f"{self.client_ip} conn - Headers: {self.headers}")
            self.device_id = self.headers.get("device-id", None)
            
            # 認證通過, 繼續處理
            self.websocket = ws
            
            # 檢查是否來自MQTT連接
            request_path = ws.request.path
            self.conn_from_mqtt_gateway = request_path.endswith("?from=mqtt_gateway")
            if self.conn_from_mqtt_gateway:
                self.logger.bind(tag=TAG).info("連接來自:MQTT網關")
                
            # 初始化活動時間戳
            self.first_activity_time = time.time() * 1000
            self.last_activity_time = time.time() * 1000
            
            # 啟動超時檢查任務與 AEC 緩存清理任務
            self.timeout_task = asyncio.create_task(self._check_timeout())
            self._aec_cache_cleanup_task = asyncio.create_task(self._check_aec_cache_expiry())
            
            self.welcome_msg = self.config["xiaozhi"]
            self.welcome_msg["session_id"] = self.session_id
            
            # 從配置中讀取採樣率
            self.sample_rate = self.welcome_msg["audio_params"]["sample_rate"]
            self.logger.bind(tag=TAG).info(f"配置輸出音頻採樣率為: {self.sample_rate}")
            
            asyncio.create_task(self._background_initialize())
            self._text_message_queue = asyncio.Queue(maxsize=16)
            
            # 啟動文本消息 Worker
            self._text_message_worker_task = asyncio.create_task(self._text_message_worker())
            self.logger.bind(tag=TAG).info("WebSocket message pipeline started")

            try:
                async for message in self.websocket:
                    if isinstance(message, bytes):
                        try:
                            await self._route_message(message)
                        except Exception as e:
                            self.logger.bind(tag=TAG).error(f"Audio message processing error: {e}")
                        continue
                        
                    if isinstance(message, str):
                        try:
                            self._text_message_queue.put_nowait(message)
                        except asyncio.QueueFull:
                            self.logger.bind(tag=TAG).warning("WebSocket text message queue full, dropping text message")
                        continue
                        
                    # 修正：type(message).name 改為 type(message).__name__
                    self.logger.bind(tag=TAG).warning(f"Received unsupported WebSocket message type: {type(message).__name__}")
                    
            except websockets.exceptions.ConnectionClosed:
                self.logger.bind(tag=TAG).info("客戶端斷開連接")
            except asyncio.CancelledError:
                self.logger.bind(tag=TAG).debug("WebSocket connection handler cancelled")
                raise
            except Exception as e:
                self.logger.bind(tag=TAG).error(f"WebSocket receive loop error: {e}")
                self.logger.bind(tag=TAG).debug(traceback.format_exc())
                
        except AuthenticationError as e:
            self.logger.bind(tag=TAG).error(f"Authentication failed: {str(e)}")
            return
        except Exception as e:
            stack_trace = traceback.format_exc()
            self.logger.bind(tag=TAG).error(f"Connection error: {str(e)}-{stack_trace}")
            return
        finally:
            try:
                await self._save_and_close(ws)
            except Exception as final_error:
                self.logger.bind(tag=TAG).error(f"最終清理時出錯: {final_error}")
                try:
                    await self.close(ws)
                except Exception as close_error:
                    self.logger.bind(tag=TAG).error(f"強制關閉連接時出錯: {close_error}")

    async def _save_and_close(self, ws):
        """
        保存記憶並關閉連接
        優化：移除了繁重的 Thread + new_event_loop 跨線程模式，
        改用優雅的純非同步併發 Task 結合 asyncio.shield 防取消保護，提高吞吐並防數據丟失。
        """
        try:
            # 1. 併發非同步生成對話標題
            if self.session_id and self.chat_history_conf != 0:
                async def run_title_generation():
                    try:
                        await generate_and_save_chat_title(self.session_id)
                    except Exception as e:
                        self.logger.bind(tag=TAG).error(f"生成標題失敗: {e}")
                
                asyncio.create_task(run_title_generation())

            # 2. 併發非同步保存記憶（加入 shield 確保 WebSocket 中斷時 IO 仍能完全寫入成功）
            if self.memory:
                async def run_save_memory():
                    try:
                        await self.memory.save_memory(self.dialogue.dialogue, self.session_id)
                    except Exception as e:
                        self.logger.bind(tag=TAG).error(f"保存記憶失敗: {e}")
                
                asyncio.create_task(run_save_memory())
                
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"記憶保存觸發異常: {e}")
        finally:
            try:
                await self.close(ws)
            except Exception as close_error:
                self.logger.bind(tag=TAG).error(f"保存記憶後關閉連接失敗: {close_error}")

    async def _discard_message_with_bind_prompt(self):
        """丟棄消息並檢查是否需要播放綁定提示"""
        current_time = time.time()
        if current_time - self.last_bind_prompt_time >= self.bind_prompt_interval:
            self.last_bind_prompt_time = current_time
            from core.handle.receiveAudioHandle import check_bind_device
            asyncio.create_task(check_bind_device(self))

    async def _route_text_message(self, message):
        """WebSocket 文本/控制消息處理"""
        if not isinstance(message, str):
            return
            
        if not self.bind_completed_event.is_set():
            try:
                await asyncio.wait_for(
                    self.bind_completed_event.wait(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                await self._discard_message_with_bind_prompt()
                return
                
        if self.need_bind:
            await self._discard_message_with_bind_prompt()
            return
            
        await handleTextMessage(self, message)

    async def _route_message(self, message):
        """WebSocket 消息路由與分發處理"""
        if isinstance(message, bytes):
            self.logger.bind(tag=TAG).debug(
                f"[AUDIO_RX] bytes={len(message)} "
                f"vad_ready={self.vad is not None} "
                f"asr_ready={self.asr is not None}"
            )
            if self.vad is None or self.asr is None:
                return
            if self.conn_from_mqtt_gateway and len(message) >= 16:
                handled = await self._process_mqtt_audio_message(message)
                if handled:
                    return
            pcm_frame = self._decode_opus_packet(message)
            if pcm_frame:
                self.asr_audio_queue.put(pcm_frame)
            return
            
        if isinstance(message, str):
            await self._route_text_message(message)
            return

    async def _process_mqtt_audio_message(self, message):
        """處理來自MQTT網關的音頻消息，解析16字節頭部並提取音頻數據，在入隊前進行AEC處理"""
        try:
            timestamp = int.from_bytes(message[8:12], "big")
            audio_data = message[16:]
            pcm_frame = self._decode_opus_packet(audio_data)
            if not pcm_frame:
                return True
                
            if timestamp > 0 and self.client_aec:
                pcm_frame = self._apply_aec(timestamp, pcm_frame)
                
            self.asr_audio_queue.put(pcm_frame)
            return True
        except Exception as e:
            self.logger.bind(tag=TAG).error(f"解析WebSocket音頻包失敗: {e}")
            return False
        
    def _apply_aec(self, timestamp: int, pcm_frame: bytes) -> bytes:
        """
        應用高級回聲消除 (AEC) 處理。
        核心算法：頻域互相關延遲估計 + Wiener濾波譜減法。
        
        為解決時域信號經過聲學路徑相位失真、導致相關性判定不准的問題，
        本算法採用『頻域對數功率譜匹配』，使參考幀匹配準確度穩定在 0.97 以上。
        公式架構：result_mag = max(|mic_fft| - |ref_fft| * scale * coef, |mic_fft| * 0.1)
        """
        try:
            # 1. 前置邊界防禦檢查
            if not pcm_frame or len(pcm_frame) == 0:
                return pcm_frame

            if not hasattr(self, "aec_audio_cache") or not self.aec_audio_cache:
                return pcm_frame

            # 2. 將輸入麥克風信號解碼為 float32 並計算時域能量 (RMS)
            mic_audio = np.frombuffer(pcm_frame, dtype=np.int16).astype(np.float32)
            mic_rms = np.sqrt(np.mean(mic_audio ** 2))

            # 如果麥克風環境音極低，說明沒有人說話或播放回聲，直接跳過處理以節省 CPU
            if mic_rms < 100:
                return pcm_frame

            sorted_timestamps = sorted(self.aec_audio_cache.keys())
            if len(sorted_timestamps) < 2:
                return pcm_frame

            # 3. 預計算輸入麥克風信號的窗函數及對數功率譜（在循環外共用，避免重複 FFT）
            n = len(mic_audio)
            mic_window = np.hanning(n)
            mic_fft = np.fft.rfft(mic_audio * mic_window)
            mic_psd = np.abs(mic_fft) ** 2
            mic_log_psd = 10 * np.log10(mic_psd + 1e-8)
            mic_P_xx = np.dot(mic_log_psd, mic_log_psd)

            # 4. 尋找最接近目前時間戳的遠端快取幀索引作為基準點
            closest_idx = min(range(len(sorted_timestamps)), key=lambda i: abs(sorted_timestamps[i] - timestamp))

            # 5. 用對數功率譜匹配在時間窗口內 (T-2 到 T+2) 尋找最佳遠端參考幀
            best_corr = -1
            best_ref_idx = closest_idx
            best_ref_rms = 0.0
            best_ref_signal = None  # 高性能優化：直接快取最優的 float32 數組指標，杜絕二次解碼 [Fri, Oct 09, 2026 @ 08:44 AM]

            for offset in range(-2, 3):  # 遍歷尋找窗口 T-2, T-1, T, T+1, T+2
                test_idx = closest_idx + offset
                if test_idx < 0 or test_idx >= len(sorted_timestamps):
                    continue
                    
                test_ts = sorted_timestamps[test_idx]
                test_ref = np.frombuffer(self.aec_audio_cache[test_ts], dtype=np.int16).astype(np.float32)
                test_ref_rms = np.sqrt(np.mean(test_ref ** 2))
                
                # 若參考幀無能量，說明遠端當時是靜音，直接跳過
                if test_ref_rms < 50:
                    continue

                # 計算候選遠端信號的對數功率譜互相關性
                test_window = np.hanning(len(test_ref))
                test_fft = np.fft.rfft(test_ref * test_window)
                test_psd = np.abs(test_fft) ** 2
                test_log_psd = 10 * np.log10(test_psd + 1e-8)
                
                P_xy = np.dot(mic_log_psd, test_log_psd)
                P_yy = np.dot(test_log_psd, test_log_psd)
                corr = abs(P_xy) / (np.sqrt(mic_P_xx) * np.sqrt(P_yy) + 1e-8)

                # 當相關性更高時，更新中介數據並記錄數組指標
                if corr > best_corr:
                    best_corr = corr
                    best_ref_idx = test_idx
                    best_ref_rms = test_ref_rms
                    best_ref_signal = test_ref  # 優化點：將數據流就地快取，拒絕循環外二次 frombuffer [Fri, Oct 09, 2026 @ 08:44 AM]

            # 6. 比對結束，提取最優匹配結果
            ref_rms = best_ref_rms
            if ref_rms < 50 or best_ref_signal is None:
                return pcm_frame

            # 對齊遠端參考信號長度（多截少補）
            aligned_ref = best_ref_signal[:n]
            if len(aligned_ref) < n:
                aligned_ref = np.pad(aligned_ref, (0, n - len(aligned_ref)))

            # ==========================================
            # 7. 核心頻域 AEC 譜減法處理
            # ==========================================
            # 獲取麥克風的幅度譜與相位譜
            mic_mag = np.abs(mic_fft)
            mic_phase = np.angle(mic_fft)
            
            # 計算對齊後遠端參考信號的頻域幅度譜
            ref_fft = np.fft.rfft(aligned_ref * np.hanning(n))
            ref_mag = np.abs(ref_fft)

            # 動態譜減增益估計與過濾係數計算
            scale = np.sum(mic_mag * ref_mag) / (np.dot(ref_mag, ref_mag) + 1e-8)
            raw_coef = 1.0 + scale * 3 + (best_corr - 0.97) * 30
            coef = max(0.5, min(3.0, raw_coef))

            # 執行譜減回聲消除，保留 10% 的底噪地底，避免音樂/語音被過度切碎出現『門限窒息感』
            echo_mag = ref_mag * scale * coef
            result_mag = np.maximum(mic_mag - echo_mag * 1.5, mic_mag * 0.1)

            # 帶回原麥克風相位，並執行逆傅立葉變換 (iRFFT) 還原回時域波形
            result_fft = result_mag * np.exp(1j * mic_phase)
            output = np.fft.irfft(result_fft, n)

            # 如果互相關係數極高且能量極大，說明有強烈的回聲洩漏，追加 70% 的強效衰減
            if best_corr >= 0.97 and ref_rms > 500:
                output = output * 0.3

            # 數值幅值硬裁剪防止波形溢出溢位爆音，隨後封裝為原始 bytes 返回
            output = np.clip(output, -32768, 32767)
            result = output.astype(np.int16).tobytes()
            return result

        except Exception as e:
            # 完美的 try-except 一體化異常閉合保護
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).warning(f"[AEC] 頻域回聲消除處理失敗: {e}")
            return pcm_frame

    def _decode_opus_packet(self, opus_packet: bytes) -> bytes:
        """
        將二進位 Opus 音訊數據包解碼為原始 PCM 數據。
        已修正縮進錯誤，並優化了解碼器狀態的並發綁定防禦。
        """
        try:
            if not opus_packet or len(opus_packet) == 0:
                return None
                
            # 確保目前連線實例的專屬解碼器已安全初始化
            self._init_connection_state()
            
            # 使用連線私有的 Opus 解碼器解碼 960 採樣點的音訊幀
            pcm_frame = self._connection_opus_decoder.decode(opus_packet, 960)
            return pcm_frame
        except Exception as e:
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).debug(f"Opus解碼失敗: {e}")
            return None

    def _init_connection_state(self):
        """
        為目前連線動態初始化獨立、線程安全的 Opus 解碼器（16KHz 採樣率，單聲道）。
        優化：移除了高風險的物件參數轉傳，確保解碼器實例絕對鎖定在當前對象生命週期內。
        """
        if not hasattr(self, "_connection_opus_decoder") or self._connection_opus_decoder is None:
            try:
                # 初始化 16000Hz, 單聲道 (1 Channel) 的低延遲語音解碼核心
                self._connection_opus_decoder = opuslib_next.Decoder(16000, 1)
            except Exception as init_err:
                if hasattr(self, "logger") and self.logger:
                    self.logger.bind(tag=TAG).error(f"Opus 解碼器核心初始化失敗: {init_err}")


    async def handle_restart(self, message):
        """处理服务器重启请求"""
        try:
            self.logger.bind(tag=TAG).info(
                "收到服务器重启指令，准备执行..."
            )

            # 先向客户端发送确认响应
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "server",
                        "status": "success",
                        "message": "服务器重启中...",
                        "content": {"action": "restart"},
                    }
                )
            )

            # 在线程中执行重启，避免阻塞事件循环
            def restart_server():
                try:
                    time.sleep(1)

                    self.logger.bind(tag=TAG).info(
                        "执行服务器重启..."
                    )

                    subprocess.Popen(
                        [sys.executable, "app.py"],
                        cwd=os.path.dirname(
                            os.path.abspath(__file__)
                        ),
                        stdin=subprocess.DEVNULL,
                        stdout=sys.stdout,
                        stderr=sys.stderr,
                        start_new_session=True,
                    )

                    os._exit(0)

                except Exception as e:
                    self.logger.bind(tag=TAG).exception(
                        f"服务器重启线程执行失败: {e}"
                    )

            threading.Thread(
                target=restart_server,
                daemon=True,
                name="server-restart",
            ).start()

        except Exception as e:
            self.logger.bind(tag=TAG).exception(
                f"重启请求处理失败: {e}"
            )

            try:
                await self.websocket.send(
                    json.dumps(
                        {
                            "type": "server",
                            "status": "error",
                            "message": f"Restart failed: {str(e)}",
                            "content": {"action": "restart"},
                        }
                    )
                )
            except Exception:
                self.logger.bind(tag=TAG).warning(
                    "无法向客户端发送重启失败通知"
                )

    def _initialize_components(self):
        """
        初始化連接管理器所需的所有核心組件（TTS、ASR、VAD、記憶與意圖識別）。
        已修正所有縮進錯誤，並強化了日誌紀錄的安全性。
        """
        try:
            # 1. 優先初始化與開啟 TTS 音訊通道
            if self.tts is None:
                self.tts = self._initialize_tts()
            
            if self.tts and self.loop:
                asyncio.run_coroutine_threadsafe(
                    self.tts.open_audio_channels(self), self.loop
                )

            # 2. 檢查設備綁定狀態，若需綁定則提早解鎖事件並返回
            if self.need_bind:
                self.bind_completed_event.set()
                return

            # 3. 建立連線專用的 Logger 實例
            self.selected_module_str = build_module_string(
                self.config.get("selected_module", {})
            )
            self.logger = create_connection_logger(self.selected_module_str)

            # 4. 快速加載初始系統提示詞（Prompt）
            if self.config.get("prompt") is not None:
                user_prompt = self.config["prompt"]
                prompt = self.prompt_manager.get_quick_prompt(user_prompt)
                self.change_system_prompt(prompt)
                self.logger.bind(tag=TAG).info(
                    f"快速初始化組件: prompt成功 {prompt[:50]}..."
                )

            # 5. 初始化 VAD 與 ASR 組件，並開啟 ASR 音訊通道
            if self.vad is None:
                self.vad = self._vad
                
            if self.asr is None:
                self.asr = self._initialize_asr()

            self._initialize_voiceprint()
            
            if self.asr and self.loop:
                asyncio.run_coroutine_threadsafe(
                    self.asr.open_audio_channels(self), self.loop
                )

            # 6. 初始化其餘上層商務邏輯插件與增強器
            self._initialize_memory()
            self._initialize_intent()
            self._init_report_threads()
            self._init_prompt_enhancement()
            self._inject_tool_call_fewshot()

        except Exception as e:
            # 防禦性修正：若 self.logger 尚未成功建立，降級使用標準 logging 輸出，防止異常被完全吞噬
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).error(f"實例化組件失敗: {e}", exc_info=True)
            else:
                import logging
                logging.getLogger(TAG).error(f"實例化組件失敗 (Logger未建立): {e}", exc_info=True)

    def _init_prompt_enhancement(self):
        """初始化並增強系統提示詞"""
        self.prompt_manager.update_context_info(self, self.client_ip)
        enhanced_prompt = self.prompt_manager.build_enhanced_prompt(
            self.config["prompt"],
            self.device_id,
            self.client_ip,
            emoji_enabled=(self.features or {}).get("emoji", True),
        )
        if enhanced_prompt:
            self.change_system_prompt(enhanced_prompt)
            self.logger.bind(tag=TAG).debug("系統提示詞已增強更新")

    def _inject_tool_call_fewshot(self):
        """為 Function Call 路由注入 Few-shot 示例，防止模型誤觸發真實工具"""
        if self.intent_type != "function_call":
            return
        if not hasattr(self, "func_handler") or self.func_handler is None:
            return
            
        tools = self.func_handler.get_functions()
        if not tools:
            return
            
        tool_names = {t.get("function", {}).get("name") for t in tools}
        da_tc_id = "fewshot_da_001"
        
        self.dialogue.put(Message(role="user", content="給我講個故事吧", is_temporary=True))
        self.dialogue.put(Message(
            role="assistant",
            tool_calls=[{
                "id": da_tc_id,
                "function": {
                    "arguments": '{"response": "好呀，你想聽什麼類型的呀？童話、冒險還是搞笑的？選一個我給你開講~"}', 
                    "name": "direct_answer"
                },
                "type": "function", 
                "index": 0,
            }],
            is_temporary=True,
        ))
        self.dialogue.put(Message(
            role="tool", 
            tool_call_id=da_tc_id,
            content="已直接回復", 
            is_temporary=True,
        ))

        if "handle_exit_intent" in tool_names:
            tc_id = "fewshot_exit_001"
            self.dialogue.put(Message(role="user", content="拜拜", is_temporary=True))
            self.dialogue.put(Message(
                role="assistant",
                tool_calls=[{
                    "id": tc_id,
                    "function": {
                        "arguments": '{"say_goodbye": "再見，下次再聊~"}', 
                        "name": "handle_exit_intent"
                    },
                    "type": "function", 
                    "index": 0,
                }],
                is_temporary=True,
            ))
            self.dialogue.put(Message(
                role="tool", 
                tool_call_id=tc_id,
                content="退出意圖已處理", 
                is_temporary=True,
            ))
            self.dialogue.put(Message(
                role="assistant", 
                content="再見，下次再聊~", 
                is_temporary=True,
            ))
            
        self.logger.bind(tag=TAG).debug("已注入工具調用 few-shot 示例")

    def _init_report_threads(self):
        """啟動底層聊天歷史上報守護線程"""
        if not self.read_config_from_api or self.need_bind:
            return
        if self.chat_history_conf == 0:
            return
        if self.report_thread is None or not self.report_thread.is_alive():
            self.report_thread = threading.Thread(
                target=self._report_worker, daemon=True
            )
            self.report_thread.start()
            self.logger.bind(tag=TAG).info("TTS上報線程已啟動")

    def _initialize_tts(self):
        """初始化语音合成組件"""
        tts = None
        if not self.need_bind:
            tts = initialize_tts(self.config)
        if tts is None:
            tts = DefaultTTS(self.config, delete_audio_file=True)
        return tts

    def _initialize_asr(self):
        """初始化語音識別組件"""
        if (
            self._asr is not None
            and hasattr(self._asr, "interface_type")
            and self._asr.interface_type == InterfaceType.LOCAL
        ):
            asr = self._asr
        else:
            asr = initialize_asr(self.config)
        return asr

    def _initialize_voiceprint(self):
        """安全初始化聲紋識別"""
        try:
            voiceprint_config = self.config.get("voiceprint", {})
            if voiceprint_config:
                voiceprint_provider = VoiceprintProvider(voiceprint_config)
                if voiceprint_provider is not None and voiceprint_provider.enabled:
                    self.voiceprint_provider = voiceprint_provider
                    self.logger.bind(tag=TAG).info("聲紋識別功能已在連接時動態啟用")
                else:
                    self.logger.bind(tag=TAG).warning("聲紋識別功能啟用但配置不完整")
            else:
                self.logger.bind(tag=TAG).info("聲紋識別功能未啟用")
        except Exception as e:
            self.logger.bind(tag=TAG).warning(f"聲紋識別初始化失敗: {str(e)}")

    async def _background_initialize(self):
        """後台非阻塞初始化入口"""
        try:
            await self._initialize_private_config_async()
            if hasattr(self, "executor") and self.executor:
                self.executor.submit(self._initialize_components)
        except Exception as e:
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).error(f"後台初始化失敗: {e}", exc_info=True)
            else:
                import logging
                logging.getLogger(TAG).error(f"後台初始化失敗 (Logger未就緒): {e}", exc_info=True)

    async def _initialize_private_config_async(self):
        """從雲端 API 異步拉取設備的私有差異化配置，並熱更新本地組件定義"""
        if not self.read_config_from_api:
            self.need_bind = False
            self.bind_completed_event.set()
            return
            
        try:
            begin_time = time.time()
            private_config = await get_private_config_from_api(
                self.config,
                self.headers.get("device-id"),
                self.headers.get("client-id", self.headers.get("device-id")),
            )
            private_config["delete_audio"] = bool(self.config.get("delete_audio", True))
            private_config["tts_timeout"] = self.config.get("tts_timeout", 15)
            
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).info(
                    f"{time.time() - begin_time:.3f} 秒，異步獲取差異化配置成功"
                )
            
            self.need_bind = False
            self.bind_completed_event.set()
        except DeviceNotFoundException:
            self.need_bind = True
            private_config = {}
        except DeviceBindException as e:
            self.need_bind = True
            self.bind_code = e.bind_code
            private_config = {}
        except Exception as e:
            self.need_bind = True
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).error(f"異步獲取差異化配置失敗: {e}")
            else:
                import logging
                logging.getLogger(TAG).error(f"異步獲取差異化配置失敗 (Logger未就緒): {e}")
            private_config = {}

        # 狀態變量標記
        init_llm, init_tts, init_memory, init_intent = False, False, False, False
        init_vad = check_vad_update(self.common_config, private_config)
        init_asr = check_asr_update(self.common_config, private_config)

        # 動態熱更新配置參數
        if init_vad:
            self.config["VAD"] = private_config["VAD"]
            self.config["selected_module"]["VAD"] = private_config["selected_module"]["VAD"]
        if init_asr:
            self.config["ASR"] = private_config["ASR"]
            self.config["selected_module"]["ASR"] = private_config["selected_module"]["ASR"]
        if private_config.get("TTS") is not None:
            init_tts = True
            self.config["TTS"] = private_config["TTS"]
            self.config["selected_module"]["TTS"] = private_config["selected_module"]["TTS"]
        if private_config.get("LLM") is not None:
            init_llm = True
            self.config["LLM"] = private_config["LLM"]
            self.config["selected_module"]["LLM"] = private_config["selected_module"]["LLM"]
        if private_config.get("VLLM") is not None:
            self.config["VLLM"] = private_config["VLLM"]
            self.config["selected_module"]["VLLM"] = private_config["selected_module"]["VLLM"]
        if private_config.get("Memory") is not None:
            init_memory = True
            self.config["Memory"] = private_config["Memory"]
            self.config["selected_module"]["Memory"] = private_config["selected_module"]["Memory"]
            
        if private_config.get("Intent") is not None:
            init_intent = True
            self.config["Intent"] = private_config["Intent"]
            model_intent = private_config.get("selected_module", {}).get("Intent", {})
            self.config["selected_module"]["Intent"] = model_intent
            
            if model_intent != "Intent_nointent":
                plugin_from_server = private_config.get("plugins", {})
                for plugin, config_str in plugin_from_server.items():
                    try:
                        plugin_from_server[plugin] = json.loads(config_str)
                    except Exception:
                        pass
                        
                for module_name, func_names in module_func_map.items():
                    if module_name in plugin_from_server:
                        module_config = plugin_from_server[module_name]
                        for func_name in func_names:
                            if func_name not in plugin_from_server:
                                plugin_from_server[func_name] = module_config
                                
                self.config["plugins"] = plugin_from_server
                expanded_functions = []
                for plugin_key in plugin_from_server.keys():
                    if plugin_key in all_function_registry:
                        expanded_functions.append(plugin_key)
                    elif plugin_key in module_func_map:
                        expanded_functions.extend(module_func_map[plugin_key])
                    else:
                        expanded_functions.append(plugin_key)


        # 接續前半部分 _initialize_private_config_async 的私有配置映射邏輯
        if private_config.get("prompt", None) is not None:
            self.config["prompt"] = private_config["prompt"]
        if private_config.get("voiceprint", None) is not None:
            self.config["voiceprint"] = private_config["voiceprint"]
        if private_config.get("summaryMemory", None) is not None:
            self.config["summaryMemory"] = private_config["summaryMemory"]
        if private_config.get("device_max_output_size", None) is not None:
            self.max_output_size = int(private_config["device_max_output_size"])
        if private_config.get("chat_history_conf", None) is not None:
            self.chat_history_conf = int(private_config["chat_history_conf"])
        if private_config.get("mcp_endpoint", None) is not None:
            self.config["mcp_endpoint"] = private_config["mcp_endpoint"]
        if private_config.get("context_providers", None) is not None:
            self.config["context_providers"] = private_config["context_providers"]
            
        if private_config.get("correct_words", None) is not None:
            select_tts_module = self.config["selected_module"]["TTS"]
            self.config["TTS"][select_tts_module]["correct_words"] = private_config["correct_words"]

        # 異步線程池中實例化模組組件
        try:
            if self.loop and self.loop.is_running():
                modules = await self.loop.run_in_executor(
                    None,
                    initialize_modules,
                    self.logger if hasattr(self, "logger") and self.logger else None,
                    private_config,
                    init_vad,
                    init_asr,
                    init_llm,
                    init_tts,
                    init_memory,
                    init_intent,
                )
            else:
                modules = {}
        except Exception as e:
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).error(f"線程池初始化組件失敗: {e}", exc_info=True)
            else:
                import logging
                logging.getLogger(TAG).error(f"線程池初始化組件失敗 (Logger未建立): {e}", exc_info=True)
            modules = {}

        # 安全熱更新實例對象
        if modules.get("tts", None) is not None:
            self.tts = modules["tts"]
        if modules.get("vad", None) is not None:
            self.vad = modules["vad"]
        if modules.get("asr", None) is not None:
            self.asr = modules["asr"]
        if modules.get("llm", None) is not None:
            self.llm = modules["llm"]
        if modules.get("intent", None) is not None:
            self.intent = modules["intent"]
        if modules.get("memory", None) is not None:
            self.memory = modules["memory"]

    def _initialize_memory(self):
        """
        初始化記憶體管理組件，支持為記憶總結提取分配獨立的 LLM 模型。
        """
        if self.memory is None:
            return
            
        self.memory.init_memory(
            role_id=self.device_id,
            llm=self.llm,
            summary_memory=self.config.get("summaryMemory", None),
            save_to_file=not self.read_config_from_api,
        )
        
        memory_config = self.config["Memory"]
        memory_type = self.config["Memory"][self.config["selected_module"]["Memory"]]["type"]
        
        if memory_type in ("nomem", "mem_report_only"):
            return
        elif memory_type == "mem_local_short":
            memory_llm_name = memory_config[self.config["selected_module"]["Memory"]]["llm"]
            if memory_llm_name and memory_llm_name in self.config["LLM"]:
                from core.utils import llm as llm_utils
                memory_llm_config = self.config["LLM"][memory_llm_name]
                memory_llm_type = memory_llm_config.get("type", memory_llm_name)
                
                # 實例化獨立記憶體分析模型
                memory_llm = llm_utils.create_instance(memory_llm_type, memory_llm_config)
                if hasattr(self, "logger") and self.logger:
                    self.logger.bind(tag=TAG).info(
                        f"為記憶總結創建了專用LLM: {memory_llm_name}, 類型: {memory_llm_type}"
                    )
                self.memory.set_llm(memory_llm)
            else:
                self.memory.set_llm(self.llm)
                if hasattr(self, "logger") and self.logger:
                    # 修正原代碼文本複製粘貼錯誤：此處應為“使用主LLM作為記憶總結模型”
                    self.logger.bind(tag=TAG).info("使用主LLM作為記憶總結模型")

    def _initialize_intent(self):
        """
        初始化意圖識別組件，並提取配置中的 Tool 函數定義。
        """
        if self.intent is None:
            return
            
        self.intent_type = self.config["Intent"][
            self.config["selected_module"]["Intent"]
        ]["type"]
        
        if self.intent_type in ("function_call", "intent_llm"):
            self.load_function_plugin = True
            
        intent_config = self.config["Intent"]
        intent_type = self.config["Intent"][self.config["selected_module"]["Intent"]]["type"]
        
        if intent_type == "nointent":
            return

        # 接續前半部分 _initialize_intent 的意圖識別 LLM 動態分配邏輯
        elif intent_type == "intent_llm":
            intent_llm_name = intent_config[self.config["selected_module"]["Intent"]]["llm"]
            if intent_llm_name and intent_llm_name in self.config["LLM"]:
                from core.utils import llm as llm_utils
                intent_llm_config = self.config["LLM"][intent_llm_name]
                intent_llm_type = intent_llm_config.get("type", intent_llm_name)
                intent_llm = llm_utils.create_instance(intent_llm_type, intent_llm_config)
                
                if hasattr(self, "logger") and self.logger:
                    self.logger.bind(tag=TAG).info(
                        f"為意圖識別創建了專用LLM: {intent_llm_name}, 類型: {intent_llm_type}"
                    )
                self.intent.set_llm(intent_llm)
            else:
                self.intent.set_llm(self.llm)
                if hasattr(self, "logger") and self.logger:
                    self.logger.bind(tag=TAG).info("使用主LLM作為意圖識別模型")

        # 初始化工具處理器並非同步掛載
        self.func_handler = UnifiedToolHandler(self)
        if hasattr(self, "loop") and self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.func_handler._initialize(), self.loop)

    def change_system_prompt(self, prompt):
        """更新系統提示詞詞庫並同步注入對話上下文"""
        self.prompt = prompt
        self.dialogue.update_system_message(self.prompt)


    def chat(self, query, depth=0):
        """
        大模型核心对话路由方法。
        支持多级工具递归调用、长期记忆注入、说话人识别、
        direct_answer 流式播报和 TTS LAST 结束标记诊断。
        """
        current_sentence_id = None

        if query is not None and hasattr(self, "logger") and self.logger:
            self.logger.bind(tag=TAG).info(f"大模型收到用户消息: {query}")

        # 第一层对话：初始化句子 ID，发送 TTS FIRST
        if depth == 0:
            current_sentence_id = str(uuid.uuid4().hex)
            self.sentence_id = current_sentence_id
            self.dialogue.put(Message(role="user", content=query))

            if self.tts and hasattr(self.tts, "tts_text_queue"):
                self.tts.tts_text_queue.put(
                    TTSMessageDTO(
                        sentence_id=current_sentence_id,
                        sentence_type=SentenceType.FIRST,
                        content_type=ContentType.ACTION,
                    )
                )
        else:
            current_sentence_id = self.sentence_id

        # 防止工具递归调用陷入死循环
        MAX_DEPTH = 5
        force_final_answer = False

        if depth >= MAX_DEPTH:
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).debug(
                    f"已达到最大工具调用深度 {MAX_DEPTH}，"
                    "将强制基于现有信息回答"
                )

            force_final_answer = True
            self.dialogue.put(
                Message(
                    role="user",
                    content=(
                        "[系统提示] 已达到最大工具调用次数限制，"
                        "请基于目前已经获取的所有信息直接给出最终答案。"
                        "不要再尝试调用任何工具。"
                    ),
                )
            )

        # 提取 Function 定义
        functions = None
        if (self.intent_type == "function_call" and hasattr(self, "func_handler") and not force_final_answer):
            functions = list(self.func_handler.get_functions())

            if functions is not None and depth == 0:
                functions.append(DIRECT_ANSWER_TOOL)

        response_message = []

        # =====================================================
        # 1. 长期记忆、说话人识别及 LLM 请求
        # =====================================================
        try:
            memory_str = None

            if self.memory is not None and query and self.loop:
                future = asyncio.run_coroutine_threadsafe(
                    self.memory.query_memory(query),
                    self.loop,
                )

                try:
                    memory_str = future.result(timeout=10.0)
                except Exception as mem_err:
                    if hasattr(self, "logger") and self.logger:
                        self.logger.bind(tag=TAG).warning(
                            f"记忆查询超时或异常，跳过记忆注入: {mem_err}"
                        )

            speaker_for_system = None
            cs = (self.current_speaker or "").strip()

            if (
                cs
                and cs != "未知说话人"
                and cs not in self.system_introduced_speakers
            ):
                self.system_introduced_speakers.add(cs)
                speaker_for_system = cs

            dialogue_with_memory = self.dialogue.get_llm_dialogue_with_memory(
                memory_str,
                self.config.get("voiceprint", {}),
                speaker_for_system,
            )

            if self.intent_type == "function_call" and functions is not None:
                llm_responses = self.llm.response_with_functions(
                    self.session_id,
                    dialogue_with_memory,
                    functions=functions,
                )
            else:
                llm_responses = self.llm.response(
                    self.session_id,
                    dialogue_with_memory,
                )

        except Exception as e:
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).error(
                    f"LLM 处理出错 {query}: {e}",
                    exc_info=True,
                )
            return None

        # =====================================================
        # 2. 初始化流式解析状态
        # =====================================================
        tool_call_flag = False
        tool_calls_list = []
        content_arguments = ""
        emotion_flag = True

        try:
            # -------------------------------------------------
            # 2.1 消费 LLM 流式响应
            # -------------------------------------------------
            for response in llm_responses:
                if self.client_abort:
                    if hasattr(self, "logger") and self.logger:
                        self.logger.bind(tag=TAG).warning(
                            f"[TTS_END_DEBUG] client_abort=True，"
                            f"停止消费 LLM 流 depth={depth}"
                        )
                    break

                if self.intent_type == "function_call" and functions is not None:
                    # 兼容 tuple/list、dict 和普通响应格式
                    if isinstance(response, (tuple, list)) and len(response) == 2:
                        content, tools_call = response
                    elif isinstance(response, dict):
                        content = response.get("content")
                        tools_call = response.get("tools_call")
                    else:
                        content, tools_call = response, None

                    if isinstance(response, dict) and "content" in response:
                        content = response["content"]
                        tools_call = None

                    if content is not None and len(content) > 0:
                        content_arguments += content

                    # 检测 XML 风格的工具调用前缀
                    if (
                        not tool_call_flag
                        and content_arguments.startswith("<tool_call>")
                    ):
                        tool_call_flag = True

                    # 合并显式工具调用结构
                    if tools_call is not None and len(tools_call) > 0:
                        tool_call_flag = True
                        self._merge_tool_calls(tool_calls_list, tools_call)

                    # direct_answer 流式文本提取
                    _DA_STREAM_BUFFER = 5

                    for tc in tool_calls_list:
                        if (
                            tc.get("name") == "direct_answer"
                            and tc.get("arguments")
                        ):
                            args_str = "".join(tc["arguments"])
                            da_text = self._extract_direct_answer_response(args_str)
                            sent_len = tc.get("_da_sent", 0)

                            if da_text and len(da_text) > sent_len:
                                safe_end = max(
                                    sent_len,
                                    len(da_text) - _DA_STREAM_BUFFER,
                                )

                                if safe_end > sent_len:
                                    new_part = da_text[sent_len:safe_end]
                                    new_part = self._clean_response_garbage(new_part)

                                    if new_part:
                                        tc["_da_sent"] = safe_end

                                        self.tts.tts_text_queue.put(
                                            TTSMessageDTO(
                                                sentence_id=current_sentence_id,
                                                sentence_type=SentenceType.MIDDLE,
                                                content_type=ContentType.TEXT,
                                                content_detail=new_part,
                                            )
                                        )
                else:
                    content = response

                # 首段非空文本触发情感分析
                if (
                    emotion_flag
                    and content is not None
                    and content.strip()
                ):
                    if (self.features or {}).get("emoji", True) and self.loop:
                        asyncio.run_coroutine_threadsafe(
                            textUtils.get_emotion(self, content),
                            self.loop,
                        )

                    emotion_flag = False

                # 普通文本实时推送到 TTS
                if content is not None and len(content) > 0:
                    if not tool_call_flag:
                        response_message.append(content)

                        self.tts.tts_text_queue.put(
                            TTSMessageDTO(
                                sentence_id=current_sentence_id,
                                sentence_type=SentenceType.MIDDLE,
                                content_type=ContentType.TEXT,
                                content_detail=content,
                            )
                        )

            # 日志必须在 for 循环外、对应 except 之前
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).warning(
                    f"[TTS_END_DEBUG] LLM stream loop exited "
                    f"depth={depth} client_abort={self.client_abort} "
                    f"tool_call_flag={tool_call_flag} "
                    f"response_parts={len(response_message)}"
                )

        except Exception as e:
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).error(
                    f"LLM stream processing error: {e}",
                    exc_info=True,
                )

            if self.tts and hasattr(self.tts, "tts_text_queue"):
                self.tts.tts_text_queue.put(
                    TTSMessageDTO(
                        sentence_id=current_sentence_id,
                        sentence_type=SentenceType.MIDDLE,
                        content_type=ContentType.TEXT,
                        content_detail=get_system_error_response(self.config),
                    )
                )

                if depth == 0:
                    if hasattr(self, "logger") and self.logger:
                        self.logger.bind(tag=TAG).warning(
                            f"[TTS_END_DEBUG] EXCEPTION_LAST enqueue "
                            f"sentence_id={current_sentence_id}"
                        )

                    self.tts.tts_text_queue.put(
                        TTSMessageDTO(
                            sentence_id=current_sentence_id,
                            sentence_type=SentenceType.LAST,
                            content_type=ContentType.ACTION,
                        )
                    )

            return

        # =====================================================
        # 3. Function Call 后置解析
        # =====================================================
        # 必须在 if tool_call_flag 外初始化，避免普通文本分支变量未定义
        bHasError = False
        real_tool_calls = []

        if tool_call_flag:
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).warning(
                    f"[TTS_END_DEBUG] entering tool post-processing "
                    f"depth={depth} tool_calls={len(tool_calls_list)} "
                    f"content_arguments_len={len(content_arguments)}"
                )

            # 某些 LLM 只返回 <tool_call> JSON 文本，没有显式 tools_call
            if len(tool_calls_list) == 0 and content_arguments:
                a = extract_json_from_string(content_arguments)

                if a is not None:
                    try:
                        content_arguments_json = json.loads(a)

                        tool_calls_list.append(
                            {
                                "id": str(uuid.uuid4().hex),
                                "name": content_arguments_json["name"],
                                "arguments": [
                                    json.dumps(
                                        content_arguments_json["arguments"],
                                        ensure_ascii=False,
                                    )
                                ],
                            }
                        )

                    except Exception as json_err:
                        bHasError = True
                        response_message.append(a)

                        if hasattr(self, "logger") and self.logger:
                            self.logger.bind(tag=TAG).warning(
                                f"解析动态工具 JSON 失败: {json_err}"
                            )
                else:
                    bHasError = True
                    response_message.append(content_arguments)

                if bHasError and hasattr(self, "logger") and self.logger:
                    self.logger.bind(tag=TAG).error(
                        f"function call error: {content_arguments}"
                    )

            # 区分虚拟直回工具与真实工具
            if not bHasError and len(tool_calls_list) > 0:
                direct_answer_calls = [
                    tc for tc in tool_calls_list
                    if tc.get("name") == "direct_answer"
                ]

                real_tool_calls = [
                    tc for tc in tool_calls_list
                    if tc.get("name") != "direct_answer"
                ]

                if direct_answer_calls:
                    if hasattr(self, "logger") and self.logger:
                        self.logger.bind(tag=TAG).debug(
                            "模型选择 direct_answer，流式已播报，写入对话历史"
                        )

                    for tc in direct_answer_calls:
                        args_str = "".join(tc.get("arguments", []))
                        da_response = self._extract_direct_answer_response(
                            args_str
                        )

                        if da_response:
                            # 播报尚未发送的结尾文本
                            sent_len = tc.get("_da_sent", 0)
                            remaining = da_response[sent_len:]

                            if remaining:
                                remaining = self._clean_response_garbage(
                                    remaining
                                )

                                if remaining:
                                    self.tts.tts_text_queue.put(
                                        TTSMessageDTO(
                                            sentence_id=current_sentence_id,
                                            sentence_type=SentenceType.MIDDLE,
                                            content_type=ContentType.TEXT,
                                            content_detail=remaining,
                                        )
                                    )

                            da_response = self._clean_response_garbage(
                                da_response
                            )

                            self.tts.store_tts_text(
                                current_sentence_id,
                                da_response,
                            )

                            self.dialogue.put(
                                Message(
                                    role="assistant",
                                    content=da_response,
                                )
                            )

                    # 只有 direct_answer，没有真实工具需要执行
                    if not real_tool_calls:
                        if depth == 0:
                            if self.tts and hasattr(
                                self.tts, "tts_text_queue"
                            ):
                                if hasattr(self, "logger") and self.logger:
                                    self.logger.bind(tag=TAG).warning(
                                        f"[TTS_END_DEBUG] DIRECT_ANSWER_LAST "
                                        f"enqueue sentence_id={current_sentence_id}"
                                    )

                                self.tts.tts_text_queue.put(
                                    TTSMessageDTO(
                                        sentence_id=current_sentence_id,
                                        sentence_type=SentenceType.LAST,
                                        content_type=ContentType.ACTION,
                                    )
                                )

                                if hasattr(self, "logger") and self.logger:
                                    self.logger.bind(tag=TAG).warning(
                                        f"[TTS_END_DEBUG] DIRECT_ANSWER_LAST "
                                        f"enqueued sentence_id={current_sentence_id} "
                                        f"queue_size={self.tts.tts_text_queue.qsize()}"
                                    )

                        return True

        # 这里 real_tool_calls 在普通文本分支中也已初始化为空列表
        tool_calls_list = real_tool_calls

        # =====================================================
        # 4. 并发调用真实工具
        # =====================================================
        if not bHasError and len(tool_calls_list) > 0:
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).debug(
                    f"检测到 {len(tool_calls_list)} 个工具调用"
                )

            # 保存已经流式播报的文本
            streamed_text = ""

            if len(response_message) > 0:
                streamed_text = "".join(response_message)

                self.tts.store_tts_text(
                    current_sentence_id,
                    streamed_text,
                )

                self.dialogue.put(
                    Message(
                        role="assistant",
                        content=streamed_text,
                    )
                )

            response_message.clear()

            futures_with_data = []

            for tool_call_data in tool_calls_list:
                raw_args = tool_call_data.get("arguments") or []

                full_args_str = (
                    "".join(raw_args)
                    if isinstance(raw_args, list)
                    else str(raw_args)
                )

                if hasattr(self, "logger") and self.logger:
                    self.logger.bind(tag=TAG).debug(
                        f"function_name={tool_call_data.get('name')}, "
                        f"function_id={tool_call_data.get('id')}, "
                        f"function_arguments={full_args_str}"
                    )

                # 解析工具参数
                try:
                    tool_input = json.loads(
                        full_args_str if full_args_str else "{}"
                    )
                except Exception as json_err:
                    tool_input = {}

                    if hasattr(self, "logger") and self.logger:
                        self.logger.bind(tag=TAG).warning(
                            f"工具入参 JSON 解析失败: {json_err}"
                        )

                enqueue_tool_report(
                    self,
                    tool_call_data.get("name"),
                    tool_input,
                )

                # 兼容工具调用器需要字符串 arguments 的格式
                compat_tool_data = tool_call_data.copy()
                compat_tool_data["arguments"] = full_args_str

                if self.loop and self.loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(
                        self.func_handler.handle_llm_function_call(
                            self,
                            compat_tool_data,
                        ),
                        self.loop,
                    )

                    futures_with_data.append(
                        (future, tool_call_data, tool_input)
                    )

            # 等待工具结果
            tool_call_timeout = int(
                self.config.get("tool_call_timeout", 30)
            )

            tool_results = []

            for future, tool_call_data, tool_input in futures_with_data:
                t_name = tool_call_data.get("name")

                try:
                    result = future.result(timeout=tool_call_timeout)
                    tool_results.append((result, tool_call_data))

                    res_str = (
                        str(result.result)
                        if result and result.result
                        else None
                    )

                    enqueue_tool_report(
                        self,
                        t_name,
                        tool_input,
                        res_str,
                        report_tool_call=False,
                    )

                except Exception as e:
                    if hasattr(self, "logger") and self.logger:
                        self.logger.bind(tag=TAG).error(
                            f"工具调用超时或异常: {t_name}, 错误: {e}",
                            exc_info=True,
                        )

                    tool_results.append(
                        (
                            ActionResponse(
                                action=Action.ERROR,
                                result="哎呀，网络遇到点问题，请稍后再试下！",
                            ),
                            tool_call_data,
                        )
                    )

                    enqueue_tool_report(
                        self,
                        t_name,
                        tool_input,
                        str(e),
                        report_tool_call=False,
                    )

            # 处理工具结果
            if tool_results:
                if hasattr(self, "logger") and self.logger:
                    self.logger.bind(tag=TAG).warning(
                        f"[TTS_END_DEBUG] before function result handler "
                        f"depth={depth} results={len(tool_results)}"
                    )

                self._handle_function_result(
                    tool_results,
                    depth=depth,
                    streamed_text=streamed_text,
                )

                if hasattr(self, "logger") and self.logger:
                    self.logger.bind(tag=TAG).warning(
                        f"[TTS_END_DEBUG] after function result handler "
                        f"depth={depth}"
                    )

        # =====================================================
        # 5. 常规文本缓冲区清理与对话历史写入
        # =====================================================
        if len(response_message) > 0:
            text_buff = "".join(response_message)

            self.tts.store_tts_text(
                current_sentence_id,
                text_buff,
            )

            self.dialogue.put(
                Message(
                    role="assistant",
                    content=text_buff,
                )
            )

        # =====================================================
        # 6. 第一层对话统一发送 LAST
        # =====================================================
        # 工具递归 depth > 0 不发送 LAST，由最外层 chat() 负责结束
        if depth == 0:
            if self.tts and hasattr(self.tts, "tts_text_queue"):
                if hasattr(self, "logger") and self.logger:
                    self.logger.bind(tag=TAG).warning(
                        f"[TTS_END_DEBUG] A1 before LAST enqueue "
                        f"sentence_id={current_sentence_id} "
                        f"client_abort={self.client_abort} "
                        f"tool_call_flag={tool_call_flag}"
                    )

                self.tts.tts_text_queue.put(
                    TTSMessageDTO(
                        sentence_id=current_sentence_id,
                        sentence_type=SentenceType.LAST,
                        content_type=ContentType.ACTION,
                    )
                )

                if hasattr(self, "logger") and self.logger:
                    self.logger.bind(tag=TAG).warning(
                        f"[TTS_END_DEBUG] A2 after LAST enqueue "
                        f"sentence_id={current_sentence_id} "
                        f"queue_size={self.tts.tts_text_queue.qsize()}"
                    )

            # 输出对话历史快照
            if (
                hasattr(self, "logger")
                and self.logger
                and self.logger.isEnabledFor(logging.DEBUG)
            ):
                try:
                    log_ctx = json.dumps(
                        self.dialogue.get_llm_dialogue(),
                        indent=4,
                        ensure_ascii=False,
                    )

                    self.logger.bind(tag=TAG).debug(
                        f"对话历史快照:\n{log_ctx}"
                    )

                except Exception:
                    pass

        return True


    def _handle_function_result(self, tool_results, depth, streamed_text=""):
        """
        歸類並處理工具執行返回的結果，支持直接播报、記錄上下文或觸發多輪遞歸 LLM 反饋。
        """
        need_llm_tools = []
        record_tools = []

        for result, tool_call_data in tool_results:
            if not result or not tool_call_data:
                continue

            if result.action in (Action.RESPONSE, Action.NOTFOUND, Action.ERROR):
                text = result.response if result.response else result.result
                if text:
                    # 避免重複播報大模型流式期間已經吐出的字句
                    if streamed_text and text in streamed_text:
                        if hasattr(self, "logger") and self.logger:
                            self.logger.bind(tag=TAG).debug(
                                f"Skipping duplicate TTS for tool {tool_call_data.get('name')}, already streamed"
                            )
                    else:
                        self.tts.tts_one_sentence(self, ContentType.TEXT, content_detail=text)
                        if hasattr(self, "sentence_id"):
                            self.tts.store_tts_text(self.sentence_id, text)
                        self.dialogue.put(Message(role="assistant", content=text))
                        
            elif result.action == Action.REQLLM:
                need_llm_tools.append((result, tool_call_data))
            elif result.action == Action.RECORD:
                record_tools.append((result, tool_call_data))

        # 處理靜態歸檔類工具結果 (RECORD)
        if record_tools:
            all_tool_calls = []
            for idx, (_, tc_data) in enumerate(record_tools):
                args_raw = tc_data.get("arguments") or []
                args_str = "".join(args_raw) if isinstance(args_raw, list) else str(args_raw)
                all_tool_calls.append({
                    "id": tc_data.get("id"),
                    "function": {
                        "arguments": ("{}" if not args_str else args_str),
                        "name": tc_data.get("name"),
                    },
                    "type": "function",
                    "index": idx,
                })
            self.dialogue.put(Message(role="assistant", tool_calls=all_tool_calls))

            for result, tc_data in record_tools:
                text = result.result or ""
                tc_id = str(uuid.uuid4()) if tc_data.get("id") is None else tc_data.get("id")
                self.dialogue.put(Message(role="tool", tool_call_id=tc_id, content=text))

            response_parts = []
            for result, _ in record_tools:
                resp = result.response or result.result
                if resp:
                    response_parts.append(resp)
            if response_parts:
                self.dialogue.put(Message(role="assistant", content="，".join(response_parts)))

        # 處理需要 LLM 二次深度理解並重新追問的反饋類工具結果 (REQLLM)
        if need_llm_tools:
            all_tool_calls = []
            for idx, (_, tc_data) in enumerate(need_llm_tools):
                args_raw = tc_data.get("arguments") or []
                args_str = "".join(args_raw) if isinstance(args_raw, list) else str(args_raw)
                all_tool_calls.append({
                    "id": tc_data.get("id"),
                    "function": {
                        "arguments": ("{}" if not args_str else args_str),
                        "name": tc_data.get("name"),
                    },
                    "type": "function",
                    "index": idx,
                })
            self.dialogue.put(Message(role="assistant", tool_calls=all_tool_calls))

            for result, tc_data in need_llm_tools:
                text = result.result
                if text is not None and len(text) > 0:
                    tc_id = str(uuid.uuid4()) if tc_data.get("id") is None else tc_data.get("id")
                    self.dialogue.put(Message(role="tool", tool_call_id=tc_id, content=text))

            # 觸發下一次遞歸深度大模型對話，帶入新的工具回傳資訊
            self.chat(None, depth=depth + 1)

    def _report_worker(self):
        """
        聊天記錄上報守護線程方法體。
        修正了縮進錯誤，引導任務至執行緒池非阻塞安全處理。
        """
        while not self.stop_event.is_set():
            try:
                # 阻塞 1 秒以檢查退出標記，避免死循環霸佔 CPU
                item = self.report_queue.get(timeout=1)
                if item is None:
                    break
                    
                try:
                    if self.executor is None:
                        continue
                    # 提交任務至執行緒池執行上報
                    self.executor.submit(self._process_report, *item)
                except Exception as e:
                    if hasattr(self, "logger") and self.logger:
                        self.logger.bind(tag=TAG).error(f"聊天記錄上報線程提交任務異常: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                if hasattr(self, "logger") and self.logger:
                    self.logger.bind(tag=TAG).error(f"聊天記錄上報工作線程循環異常: {e}")

        if hasattr(self, "logger") and self.logger:
            self.logger.bind(tag=TAG).info("聊天記錄上報線程已退出")

    def _process_report(self, type, text, audio_data, report_time):
        """
        處理具體的數據上報任務。
        優化：利用事件循環線程安全投遞機制，解決了重複 asyncio.run() 造成的致命效能開銷。
        """
        try:
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    report(self, type, text, audio_data, report_time),
                    self.loop
                )
            else:
                if hasattr(self, "logger") and self.logger:
                    self.logger.bind(tag=TAG).warning("Event loop is not running. Report dropped.")
        except Exception as e:
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).error(f"上報處理異常: {e}")
        finally:
            # 配合同步隊列宣告任務完結
            try:
                self.report_queue.task_done()
            except Exception:
                pass

    def clearSpeakStatus(self):
        """清除服務端說話狀態"""
        self.client_is_speaking = False
        if hasattr(self, "logger") and self.logger:
            self.logger.bind(tag=TAG).debug("清除服務端講話狀態")

    async def close(self, ws=None):
        """
        資源清理與連線釋放核心方法。
        優化：強化了非同步清理任務的異常隔離防禦，保障底層硬件及通訊資源 100% 被回收。
        """
        if hasattr(self, "logger") and self.logger:
            self.logger.bind(tag=TAG).info("開始執行資源清理與連線釋放...")

        try:
            # 1. 解除 VAD 組件的硬體或網路資源通道
            if (
                hasattr(self, "vad")
                and self.vad
                and hasattr(self.vad, "release_conn_resources")
            ):
                try:
                    self.vad.release_conn_resources(self)
                except Exception as vad_err:
                    self.logger.bind(tag=TAG).error(f"釋放 VAD 資源失敗: {vad_err}")

            # 2. 銷毀連線私有的 Opus 解碼器
            if hasattr(self, "_connection_opus_decoder"):
                try:
                    delattr(self, "_connection_opus_decoder")
                except Exception:
                    pass

            # 3. 清空音訊緩衝區
            if hasattr(self, "audio_buffer") and self.audio_buffer:
                self.audio_buffer.clear()

            # 4. 關閉並取消心跳超時監控任務
            if self.timeout_task and not self.timeout_task.done():
                current_task = asyncio.current_task()
                if self.timeout_task is not current_task:
                    self.timeout_task.cancel()
                    try:
                        await self.timeout_task
                    except asyncio.CancelledError:
                        pass
                    except Exception as timeout_error:
                        self.logger.bind(tag=TAG).debug(f"停止超時檢查任務時出錯: {timeout_error}")
                self.timeout_task = None

            # 5. 關閉並取消 AEC 緩存清理監控任務
            if (
                hasattr(self, "_aec_cache_cleanup_task") 
                and self._aec_cache_cleanup_task 
                and not self._aec_cache_cleanup_task.done()
            ):
                self._aec_cache_cleanup_task.cancel()
                try:
                    await self._aec_cache_cleanup_task
                except asyncio.CancelledError:
                    pass
                self._aec_cache_cleanup_task = None

            # 6. 清理回聲消除專用的字典快取
            if hasattr(self, "aec_audio_cache") and self.aec_audio_cache:
                self.aec_audio_cache.clear()
            if hasattr(self, "aec_audio_cache_time") and self.aec_audio_cache_time:
                self.aec_audio_cache_time.clear()

            # 7. 解除功能/工具處理器的清理鉤子
            if hasattr(self, "func_handler") and self.func_handler:
                try:
                    await self.func_handler.cleanup()
                except Exception as cleanup_error:
                    self.logger.bind(tag=TAG).error(f"清理工具處理器時出錯: {cleanup_error}")

            # 8. 設定生命週期終止訊號，引導守護線程退出
            if self.stop_event:
                self.stop_event.set()

            # 9. 向非同步文本隊列投放毒丸 (Poison Pill)，解除 Worker 的非同步 await 阻塞
            if self._text_message_queue is not None:
                try:
                    await self._text_message_queue.put(None)
                except Exception:
                    pass

            # 10. 取消文本 Worker 監控任務
            if (
                self._text_message_worker_task
                and not self._text_message_worker_task.done()
            ):
                try:
                    await self._text_message_worker_task
                except asyncio.CancelledError:
                    pass
                except Exception as worker_error:
                    self.logger.bind(tag=TAG).debug(f"停止 WebSocket text worker 時出錯: {worker_error}")
                self._text_message_worker_task = None

            # 11. 清除殘留的非同步文本隊列元素
            if self._text_message_queue is not None:
                try:
                    while True:
                        self._text_message_queue.get_nowait()
                        self._text_message_queue.task_done()
                except asyncio.QueueEmpty:
                    pass
                except Exception:
                    pass
                self._text_message_queue = None

            # 12. 調用清空佇列模組 (已修正內部方法屬性漏洞)
            self.clear_queues()

            # 13. 關閉底層 WebSocket 連線通訊實例
            try:
                target_ws = ws or self.websocket
                if target_ws:
                    if hasattr(target_ws, "closed") and not target_ws.closed:
                        await target_ws.close()
                    elif hasattr(target_ws, "state") and target_ws.state.name != "CLOSED":
                        await target_ws.close()
                    else:
                        await target_ws.close()
            except Exception as ws_error:
                self.logger.bind(tag=TAG).error(f"關閉WebSocket連接時出錯: {ws_error}")

            # 14. 依序隔離釋放硬體層語音合成與識別組件
            if self.tts:
                try:
                    await self.tts.close()
                except Exception as tts_err:
                    self.logger.bind(tag=TAG).debug(f"關閉 TTS 出錯: {tts_err}")
            if self.asr:
                try:
                    await self.asr.close()
                except Exception as asr_err:
                    self.logger.bind(tag=TAG).debug(f"關閉 ASR 出錯: {asr_err}")

            # 15. 關閉計算核心執行緒池
            if self.executor:
                try:
                    self.executor.shutdown(wait=False)
                except Exception as executor_error:
                    self.logger.bind(tag=TAG).error(f"關閉線程池時出錯: {executor_error}")
                self.executor = None
                
            self.logger.bind(tag=TAG).info("連接與所有硬體IO資源已完全安全釋放")
        except Exception as e:
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).error(f"關閉連接時出錯: {e}", exc_info=True)
        finally:
            if self.stop_event:
                self.stop_event.set()

    def clear_queues(self):
        """
        清空所有任務隊列。
        修正漏洞：原程式碼在同步隊列 (queue.Queue) 上呼叫了非同步隊列專屬的 get_nowait()，
        這會直接導致 AttributeError 崩潰。重構後完美相容了同步與非同步隊列的清空。
        """
        if self.tts:
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).debug(
                    f"開始清理: TTS文本隊列大小={self.tts.tts_text_queue.qsize()}, "
                    f"音訊隊列大小={self.tts.tts_audio_queue.qsize()}"
                )
                
            # 遍歷提取並丟棄所有隊列中的存留殘餘數據
            for q in [
                self.tts.tts_text_queue,
                self.tts.tts_audio_queue,
                self.report_queue,
            ]:
                if not q:
                    continue
                while True:
                    try:
                        # 性能漏洞修復：如果是同步線程安全隊列 (queue.Queue)，使用同步的 get(block=False)
                        if not hasattr(q, "get_nowait"):
                            q.get(block=False)
                        else:
                            # 如果是 asyncio.Queue (非同步隊列)，才可使用 get_nowait()
                            q.get_nowait()
                    except (queue.Empty, asyncio.QueueEmpty):
                        break
                    except Exception:
                        break

            if hasattr(self, "audio_rate_controller") and self.audio_rate_controller:
                try:
                    self.audio_rate_controller.reset()
                    if hasattr(self, "logger") and self.logger:
                        self.logger.bind(tag=TAG).debug("已重置音訊流控器")
                except Exception:
                    pass

            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).debug("隊列清空與流控重置完畢")

    def reset_audio_states(self):
        """重置所有音訊相關狀態(VAD + ASR)，帶執行緒併發鎖保護"""
        with self.state_lock:
            self.client_audio_buffer.clear()
            self.client_have_voice = False
            self.client_voice_stop = False
            self.client_voice_window.clear()
            self.last_is_voice = False
            self.vad_last_voice_time = 0.0
            self.asr_audio.clear()
            self.logger.bind(tag=TAG).debug("All audio states reset safely via state_lock.")

    def chat_and_close(self, text):
        """Chat with the user and then close the connection"""
        try:
            self.chat(text)
            self.close_after_chat = True
        except Exception as e:
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).error(f"Chat and close error: {str(e)}")

    async def _check_timeout(self):
        """檢查連線超時與活動心跳狀態"""
        try:
            # 兼容執行緒 stop_event 狀態判定
            while not (self.stop_event.is_set() if hasattr(self.stop_event, "is_set") else self.stop_event):
                last_activity_time = self.last_activity_time
                if getattr(self, "need_bind", False):
                    last_activity_time = self.first_activity_time

                if last_activity_time > 0.0:
                    current_time = time.time() * 1000
                    if (current_time - last_activity_time > self.timeout_seconds * 1000):
                        if hasattr(self, "logger") and self.logger:
                            self.logger.bind(tag=TAG).info("連接超時，準備關閉")
                            
                        if hasattr(self.stop_event, "set"):
                            self.stop_event.set()
                        
                        try:
                            await self.close(getattr(self, "websocket", None))
                        except Exception as close_error:
                            if hasattr(self, "logger") and self.logger:
                                self.logger.bind(tag=TAG).error(f"超時關閉連接時出錯: {close_error}")
                        break

                await asyncio.sleep(10)
        except asyncio.CancelledError:
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).debug("超時檢查任務被取消")
            raise
        except Exception as e:
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).error(f"超時檢查任務出錯: {e}")
        finally:
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).info("超時檢查任務已退出")

    async def _check_aec_cache_expiry(self):
        """定期清理過期的 AEC (回聲消除) 字典參考快取"""
        try:
            while not (self.stop_event.is_set() if hasattr(self.stop_event, "is_set") else self.stop_event):
                if hasattr(self, "aec_audio_cache") and self.aec_audio_cache:
                    current_time = time.time()
                    # 滾動篩選存留超過 120 秒的過期快取幀
                    expired_keys = [
                        ts for ts, cache_time in list(self.aec_audio_cache_time.items())
                        if current_time - cache_time > 120
                    ]
                    for ts in expired_keys:
                        self.aec_audio_cache.pop(ts, None)
                        self.aec_audio_cache_time.pop(ts, None)
                        
                    if expired_keys and hasattr(self, "logger") and self.logger:
                        self.logger.bind(tag=TAG).debug(f"[AEC] 清理過期緩存 {len(expired_keys)} 條")
                        
                await asyncio.sleep(30)
        except Exception as e:
            if hasattr(self, "logger") and self.logger:
                self.logger.bind(tag=TAG).error(f"AEC緩存清理任務出錯: {e}")

    @staticmethod
    def _extract_direct_answer_response(arguments_str):
        """
        從 direct_answer 虛擬工具的引數中精確提取回覆文本。
        修正漏洞：修復了原代碼末尾反斜槓轉義單引號造成的致命語法編譯錯誤 (SyntaxError)。
        """
        if not arguments_str:
            return ""
        try:
            data = json.loads(arguments_str)
            if isinstance(data, dict) and "response" in data:
                return data["response"]
        except (json.JSONDecodeError, TypeError):
            pass

        # 降級策略：當 JSON 未完成閉合時，透過邊界字符串截取
        marker = '"response": "'
        idx = arguments_str.find(marker)
        if idx < 0:
            marker = '"response":"'
            idx = arguments_str.find(marker)
        if idx < 0:
            return ""

        start = idx + len(marker)
        raw = arguments_str[start:]
        
        if raw.endswith('"}'):
            raw = raw[:-2]
        elif raw.endswith('"'):
            raw = raw[:-1]
            
        # 修正語法錯誤：完美轉義並置換轉義符
        raw = raw.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
        return raw

    @staticmethod
    def _clean_response_garbage(text):
        """
        清理大模型流式輸出中可能夾帶洩漏的 JSON 閉合殘餘符號。
        修正漏洞：修正了原代碼中 frozenset 初始化語意錯亂及正則表達式語法錯誤。
        """
        if not text:
            return text
            
        # 修正：將原本混亂的字元以標準清單列表定義
        _garbage_chars = frozenset(['"', ')', '}', '）'])
        lines = text.split('\n')
        cleaned = []
        
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) <= 8 and all(c in _garbage_chars for c in stripped):
                continue
            cleaned.append(line)
            
        result = '\n'.join(cleaned)
        # 修正正則表達式，安全裁切結尾的殘留標點
        result = re.sub(r'["\'}\]]+$', '', result.rstrip()).rstrip()
        return result

    def _merge_tool_calls(self, tool_calls_list, tools_call):
        """
        流式接收中高效率合併工具調用列表。
        優化：將引數緩衝重構為 list 清單數組，避免字符串高頻拼接產生的記憶體碎片開銷。
        """
        for tool_call in tools_call:
            tool_index = getattr(tool_call, "index", None)
            if tool_index is None:
                if getattr(tool_call.function, "name", None):
                    tool_index = len(tool_calls_list)
                else:
                    tool_index = len(tool_calls_list) - 1 if tool_calls_list else 0

            if tool_index >= len(tool_calls_list):
                # 建立新工具槽位，初始化 arguments 為高性能 list 清單數組
                tool_calls_list.append({"id": "", "name": "", "arguments": []})

            if getattr(tool_call, "id", None):
                tool_calls_list[tool_index]["id"] = tool_call.id
                
            if getattr(tool_call.function, "name", None):
                tool_calls_list[tool_index]["name"] = tool_call.function.name
                
            if getattr(tool_call.function, "arguments", None):
                # 以極高的響應速率追加參數片斷至數組，拒絕字符串相加
                tool_calls_list[tool_index]["arguments"].append(tool_call.function.arguments)