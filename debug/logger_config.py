import logging
import os
import sys

LOGGER_NAME = 'APP_LOGGER'

def get_logger():
    dbg = logging.getLogger(LOGGER_NAME)
    dbg.setLevel(logging.INFO)
    dbg.propagate = False

    if not dbg.handlers:
        log_file = os.environ.get('LOG_FILE_PATH')
        if not log_file:
            home_dir = os.path.expanduser('~')
            log_dir = os.path.join(home_dir, 'var_logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'app.log')

        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] (%(name)s) [%(filename)s:%(lineno)d]: %(message)s'
        )

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        dbg.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        dbg.addHandler(stream_handler)

        dbg.info(f"Custom logger '{LOGGER_NAME}' initialized. Log file path: {log_file}")

    return dbg

# Usage: from logger_config import dbg
# dbg.info("Your message")

dbg = get_logger()
