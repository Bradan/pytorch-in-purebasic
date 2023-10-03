import torch
import torchvision


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, 1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(8, 16, 3, 1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, 3, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, 3, 1),
            torch.nn.LeakyReLU()
        )

        self.linear = torch.nn.Linear(32, 10)

    def forward(self, x: torch.Tensor):
        while x.dim() < 4:
            x = x.unsqueeze(0)

        x = self.layers(x)

        # 32 features to 10 features
        x = x.view(x.size(0), -1)

        # softmax along the last dimension
        x = torch.nn.functional.softmax(self.linear(x), dim=-1)

        return x


if __name__ == '__main__':
    model = Model()
    x_in = torch.randn((1, 1, 28, 28))
    x_out = model(x_in)
    print(x_out)
