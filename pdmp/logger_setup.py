# logger.py
import logging
import sys
import os


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create the console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel("WARNING")

    # Create a formatter
    formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M')
    console_handler.setFormatter(formatter)

    # Check if the logger already has handlers. If the logger doesn't have any
    # handlers, add the new handler and set propagate to False to prevent the
    # log messages from being passed to the root logger and possibly being
    # duplicated.

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.propagate = False

    return logger


def setup_file_handler(logger: logging.Logger,
                       log_dir: str,
                       log_file: str = "mcmc_run.log",
                       level="INFO"):

    # create log dir and set log path
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # remove old log-file if exists
    if os.path.exists(log_path):
        os.remove(log_path)

    # set up file_handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    if logger.hasHandlers():
        formatter = logger.handlers[0].formatter
    else:
        formatter = logging.Formatter(
            '[%(asctime)s][%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Custom logging initialized. Log file: %s", log_path)
