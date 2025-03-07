from logging.handlers import RotatingFileHandler
import logging
import os
import datetime


class Logger:
    """
    Provides a logger for informative print statements and saves them for further investigation
    """
    _logger = None

    @staticmethod
    def setup_logger():
        if Logger._logger is None:
            Logger._logger = logging.getLogger()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

            os.makedirs(os.path.join("logs", ), exist_ok=True)

            file_handler = RotatingFileHandler(
                os.path.join(
                    "logs",
                    f"{str(datetime.datetime.now()).replace('-', '_').replace(':', '_')}.log",
                ),
                mode="a",
                maxBytes=10 * 1024 * 1024,
                backupCount=2,
                encoding="utf-8",
                delay=False,
            )
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            Logger._logger.setLevel(logging.INFO)
            Logger._logger.addHandler(file_handler)
            Logger._logger.addHandler(console_handler)
        return Logger._logger
