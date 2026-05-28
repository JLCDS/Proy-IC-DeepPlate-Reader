import sys
from loguru import logger as _logger


def get_logger(name: str = "deepplate"):
    _logger.remove()
    _logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> - {message}",
        level="INFO",
        colorize=True,
    )
    _logger.add(
        "logs/{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
    )
    return _logger.bind(name=name)
