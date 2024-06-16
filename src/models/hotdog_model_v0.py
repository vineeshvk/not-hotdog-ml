from torch import nn


class HotdogModelV0(nn.Module):
    def __init__(self, in_c: int = 3, out_c: int = 1, hidden_u: int = 10) -> None:
        super().__init__()

        self.convo_layer1 = nn.Sequential(
            nn.Conv2d(
                in_c,
                hidden_u,
                2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.convo_layer2 = nn.Sequential(
            nn.Conv2d(hidden_u, hidden_u, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.process_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_u * 31 * 31, out_c),
        )

    def forward(self, X):
        X = self.convo_layer1(X)
        X = self.convo_layer2(X)
        return self.process_layer(X)
