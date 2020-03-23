import logging
import sys


def config_logging(log_level, file_name):
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    log_formatter = logging.Formatter(fmt='%(asctime)s_%(name)s_%(levelname)s: %(message)s')

    file_handler = logging.FileHandler("logs/" + file_name, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    sysout_handler = logging.StreamHandler(stream=sys.stdout)
    sysout_handler.setFormatter(log_formatter)
    root_logger.addHandler(sysout_handler)
