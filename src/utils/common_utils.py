from os import path
from pathlib import Path
import requests
from zipfile import ZipFile

from src.utils.logger import logging
from src.utils.constants import DATASET_DIR_PATH


class CommonUtils:
    def download_data(self, url: str, filepath: str, force=False):
        if Path(filepath).is_file() and not force:
            return filepath
        else:
            request = requests.get(url)

        with open(filepath, "wb") as f:
            f.write(request.content)
        return filepath

    def zipFile(self, filepath: str, extract_to: str):
        if Path(filepath).is_file():
            with ZipFile(filepath, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            raise Exception("Error: File not found")
