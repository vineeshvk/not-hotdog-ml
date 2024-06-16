import torch
from torch.optim import Adam
from torch import nn
from src.data.data_ingestion import DataIngestion
from src.models.hotdog_model_v0 import HotdogModelV0
from src.utils.logger import logging
from torchmetrics.functional import accuracy


class TrainPipeline:
    def __init__(self) -> None:
        self.data_loader = DataIngestion()
        self.train_dataset, self.test_dataset = self.data_loader.get_data()

        self.model = HotdogModelV0()

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(params=self.model.parameters(), lr=0.01)

    def start_training(self):
        try:
            self._train_loop(epoch=1)
            self.get_test_stats()

        except Exception as e:
            logging.error(e)

    def get_test_stats(self):
        self.model.eval()
        with torch.inference_mode():
            t_loss, t_acc = 0, 0
            for batch, (X, y) in enumerate(self.test_dataset):
                y_pred = self.model(X).squeeze()
                t_loss += self.loss_fn(y_pred, y.to(torch.float))
                t_acc += accuracy(y_pred, y, "binary")
        t_batches = len(self.test_dataset)

        logging.info(f"Testing result -> l: {t_loss/ t_batches}, a: {t_acc/t_batches}")

    def _train_loop(self, epoch: int):
        for x in range(epoch):
            t_loss, t_acc = 0, 0

            for batch, (X, y) in enumerate(self.train_dataset):
                loss, acc = self._train_step(X,y)
                t_loss += loss
                t_acc += acc
            t_batches = len(self.train_dataset)

            logging.info(
                f"Epoch {epoch + 1} -> l: {t_loss/ t_batches}, a: {t_acc/ t_batches}"
            )

    def _train_step(self, X: torch.Tensor, y: torch.Tensor):
        self.model.train()

        y_pred: torch.Tensor = self.model(X).squeeze()

        loss: torch.Tensor = self.loss_fn(y_pred, y.to(torch.float))

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        return loss, accuracy(y_pred, y, task="binary")
