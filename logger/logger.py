import logging
import sys

APP_LOGGER_NAME = 'EssayEvaluation'

def setup_applevel_logger(logger_name=APP_LOGGER_NAME, file_name=None):
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    #Commented, since log messages will be shown because of the file handler
    # # Add a stream handler to print log messages to the console
    # sh = logging.StreamHandler(sys.stdout)
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    # If a file name is provided, add a file handler to write log messages to the file
    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def get_logger(module_name):
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)

