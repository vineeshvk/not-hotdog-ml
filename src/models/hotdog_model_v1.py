from torch import nn
from torchvision.models import resnet18


class HotdogModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, X):
        return self.resnet(X)
