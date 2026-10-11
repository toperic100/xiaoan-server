import asyncio
from loguru import logger

async def get_weather(*args, **kwargs) -> str:
    """
    🎯 终极全异步防卡死空天气插件
    1. 使用 'async def' 声明，完美契合服务端的 'await' 表达式要求。
    2. 使用 *args, **kwargs 接收任何形式的传参，彻底杜绝重复传参崩溃。
    """
    location = "广州"
    if args:
        location = args[0]
    elif "location" in kwargs:
        location = kwargs["location"]
        
    logger.bind(tag="WEATHER_PATCH").info(f"全异步天气插件成功拦截调用，目标城市: {location}")
    
    # 0 毫秒极速返回标准字符串，满足服务端的 await 表达式期望
    return f"{location}今天天气晴朗，温度舒适，非常适合出门哦。"

# 显式导出，确保小智的统一工具处理器顺畅识别
__all__ = ["get_weather"]
