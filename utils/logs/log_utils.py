import sys

from loguru import logger


def init_logger():
    logger.remove()
    logger.add(sys.stdout)


def get_logger():
    return logger
