import importlib
import pkgutil
import sys
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

# 已经完成扫描/加载的插件包
_loaded_packages = set()


def auto_import_modules(package_name):
    """
    自动加载指定插件包下的所有模块。

    特点：
    1. 同一个 package 只扫描一次
    2. 已经存在于 sys.modules 的模块不重复 import
    3. 保持原有插件注册机制不变
    """

    # ---------------------------------------------------------
    # 1. 防止同一个插件包被重复扫描
    # ---------------------------------------------------------
    if package_name in _loaded_packages:
        logger.debug(
            f"[PLUGIN] package already loaded, skip scan: {package_name}"
        )
        return

    try:
        # -----------------------------------------------------
        # 2. 导入插件 package
        # -----------------------------------------------------
        package = importlib.import_module(package_name)

        package_path = package.__path__

        loaded_count = 0
        skipped_count = 0

        # -----------------------------------------------------
        # 3. 扫描 package 下的模块
        # -----------------------------------------------------
        for _, module_name, _ in pkgutil.iter_modules(package_path):

            full_module_name = f"{package_name}.{module_name}"

            # -------------------------------------------------
            # 4. Python 已经加载过的模块，不再重复 import
            # -------------------------------------------------
            if full_module_name in sys.modules:
                skipped_count += 1
                continue

            try:
                importlib.import_module(full_module_name)
                loaded_count += 1

                logger.debug(
                    f"[PLUGIN] loaded module: {full_module_name}"
                )

            except Exception as e:
                logger.error(
                    f"[PLUGIN] failed to load module "
                    f"{full_module_name}: {e}"
                )

        # -----------------------------------------------------
        # 5. 标记 package 已经完成扫描
        # -----------------------------------------------------
        _loaded_packages.add(package_name)

        logger.info(
            f"[PLUGIN] package loaded: {package_name}, "
            f"new_modules={loaded_count}, "
            f"already_loaded={skipped_count}"
        )

    except Exception as e:
        logger.error(
            f"[PLUGIN] failed to load package "
            f"{package_name}: {e}"
        )