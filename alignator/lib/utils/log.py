import logging
import os


def setup_logging():
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] [%(threadName)s] - %(message)s",
        level=log_level,
    )
