import pickle
import torch
from torch import nn
from src.data.data_transformer import DataTransformer


class PredictPipeline:

    def __init__(self, image) -> None:
        self.tranformer = DataTransformer()

        self.model: nn.Module = torch.load("artifacts/models/model_v1.pt")
        self.image = image

    def predict(self):
        img_bytes = self.image.read()
        image_tensor = self.tranformer.convert(img_bytes)
        self.model.eval()

        with torch.no_grad():
            result = self.model(image_tensor).squeeze()
            prob = torch.sigmoid(result)
            predicted = (prob > 0.5).float()

            print([result, predicted, round(prob.item(), 1), prob, prob > 0.5])

        return predicted.item()
