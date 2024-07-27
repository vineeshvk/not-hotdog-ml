import torch
from src.data.data_transformer import DataTransformer
from src.utils.common_utils import CommonUtils
from src.utils.logger import logging

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

from os import listdir
from os.path import isfile, join


from src.utils.constants import (
    DATA_DOWNLOAD_URL,
    DATASET_DIR_PATH,
    DATASET_EXTRACTED_DIR,
    HOTDOG_DATASET_ZIP_FILE_PATH,
)


class DataIngestion:
    def __init__(self):
        self.utils = CommonUtils()
        self.download_url = DATA_DOWNLOAD_URL
        self.download_save_path = HOTDOG_DATASET_ZIP_FILE_PATH
        self.extracted_file_path = DATASET_EXTRACTED_DIR
        self.transformer = DataTransformer()

    def get_data(self):
        # self.download_and_extract()
        train_dataset = self.create_dataset(dir=self.extracted_file_path + "train")
        test_dataset = self.create_dataset(dir=self.extracted_file_path + "holdout")

        return train_dataset, test_dataset

    def download_and_extract(self):
        filePath = self.utils.download_data(
            url=self.download_url,
            filepath=self.download_save_path,
        )

        self.utils.zipFile(filePath, DATASET_DIR_PATH)

    def create_dataset(self, dir: str) -> DataLoader[torch.Tensor]:

        dataset = datasets.ImageFolder(root=dir, transform=self.transformer.transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

        # Changing the hot dog mapping as it is loaded in reverse
        class_mapping = {"not_hot_dog": 0, "hot_dog": 1}
        dataset.class_to_idx = class_mapping
        dataset.classes = ["not_hot_dog", "hot_dog"]

        return dataloader
