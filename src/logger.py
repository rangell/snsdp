import logging
import os
import time
from datetime import timedelta


class LogFormatter(logging.Formatter):
    def __init__(self):
        self.start_time = time.time()

    def format(self, record) -> str:
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def create_logger(output_dir: str, debug: bool) -> logging.Logger:
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if not debug:
        file_handler = logging.FileHandler(os.path.join(output_dir, "out.log", "a"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not debug:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
