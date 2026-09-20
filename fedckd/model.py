# Adapted from PFLlib (Apache-2.0); base class and hidden output modified.
# See THIRD_PARTY_NOTICES.md.
import torch


class FashionCNN(torch.nn.Module):
    """
    CNN architecture adapted from the PFLlib FedAvgCNN backbone.
    Used as the FedCKD backbone for Fashion-MNIST.

    Architecture:
    - Conv1: in_channels -> 32, kernel_size=5, padding=0, stride=1 + ReLU + MaxPool(2,2)
    - Conv2: 32 -> 64, kernel_size=5, padding=0, stride=1 + ReLU + MaxPool(2,2)
    - FC1: dim -> 512 + ReLU
    - FC2: 512 -> num_classes

    For 28x28 Fashion-MNIST input: dim=1024
    """

    def __init__(self, in_channels, num_classes, dim=1024):
        super(FashionCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dim = dim

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, 32, kernel_size=5, padding=0, stride=1, bias=True
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(dim, 512), torch.nn.ReLU(inplace=True)
        )
        self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, x, return_hidden=False):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        hidden = self.fc1(out)
        out = self.fc(hidden)
        if return_hidden:
            return out, hidden
        return out
