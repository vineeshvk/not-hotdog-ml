import logging
import os
from datetime import datetime

LOG_FILE_NAME = datetime.now().strftime(r"%Y_%m_%d.log")
LOG_FOLDER_PATH = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_FOLDER_PATH, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_FOLDER_PATH, LOG_FILE_NAME)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
