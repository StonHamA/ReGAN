import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.res50 = torchvision.models.resnet50()

    def forward(self, x):
        x = self.res50(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = Net().to(device)

summary(model, (3, 384, 192))