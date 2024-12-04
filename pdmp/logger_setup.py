# logger.py
import logging
import sys


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create the console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel("WARNING")

    # Create a formatter
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M')
    console_handler.setFormatter(formatter)

    # Check if the logger already has handlers. If the logger doesn't have any
    # handlers, add the new handler and set propagate to False to prevent the
    # log messages from being passed to the root logger and possibly being
    # duplicated.

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.propagate = False

    return logger

def setup_file_handler(fname, logger, level="INFO"):
    # set up file_handler
    file_handler = logging.FileHandler(fname)
    file_handler.setLevel(level)
    if logger.hasHandlers():
        formatter = logger.handlers[0].formatter
    else:
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
