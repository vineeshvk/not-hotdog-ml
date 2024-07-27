import io
from torchvision import transforms
import torch
from PIL import Image


class DataTransformer:
    def __init__(self) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )

    def convert(self, image_bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(image_bytes))
        return self.transform(image).unsqueeze(0)
