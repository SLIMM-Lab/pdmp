# logger.py
import logging
import sys
import os


def setup_logger(name: str) -> logging.Logger:
    """Setting up the logger.

    Args:
        name: name of the logger that is displayed in the log messages.

    Returns:
        logging.logger: a logger object that can be used to log messages.
    """
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
                       level="INFO",
                       append: bool = False):
    """Setting up the file handler for the logger.

    Args:
        logger: The logger object to which the file handler will be added.
        log_dir: The directory where the log file will be created.
        log_file: The name of the log file. Defaults to "mcmc_run.log".
        level: The logging level for the file handler. Defaults to "INFO".
        append: If True, keep any existing log file and append to it (used when
            resuming an interrupted run); if False, overwrite it (a fresh
            start). Defaults to False.
    """

    # create log dir and set log path
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # On a fresh start, remove any old log; on resume, keep it and append so the
    # log file is continuous across the chain of restarts.
    if not append and os.path.exists(log_path):
        os.remove(log_path)

    # set up file_handler
    file_handler = logging.FileHandler(log_path, mode='a' if append else 'w')
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


def suppress_external_loggers():
    """Suppress verbose output from external libraries.

    This function sets the logging level of external libraries to WARNING
    to prevent them from cluttering the console output with INFO/DEBUG messages,
    while still allowing important warnings and errors to be displayed.

    Currently suppresses:
        - jax_fem: Finite element solver library

    Note: This should be called AFTER the external library has been imported,
    so that its handlers are already set up.
    """
    # Get the jax_fem logger (will be imported when forward model is created)
    jax_fem_logger = logging.getLogger('jax_fem')

    # Set the logger level to WARNING
    jax_fem_logger.setLevel(logging.WARNING)

    # Also set all its handlers to WARNING level
    for handler in jax_fem_logger.handlers:
        handler.setLevel(logging.WARNING)
