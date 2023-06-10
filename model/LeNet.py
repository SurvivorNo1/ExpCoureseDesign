import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 模型结构
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                               kernel_size=5)  # 28*28*1 -> 24*24*6
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # 24*24*6 -> 12*12*6
        self.conv3 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5)  # 12*12*6 -> 8*8*16
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)  # 8*8*16 -> 4*4*16
        self.fc5 = nn.Linear(in_features=4*4*16,
                             out_features=120)  # 4*4*16 -> 120
        self.fc6 = nn.Linear(in_features=120, out_features=84)  # 120 -> 84
        self.out = nn.Linear(in_features=84, out_features=10)  # 84 -> 10

    def forward(self, X):
        # C1
        X = self.conv1(X)
        # S2
        X = self.maxpool2(X)
        X = F.relu(X)
        # C3
        X = self.conv3(X)
        # S4
        X = self.maxpool4(X)
        X = F.relu(X)
        # flatten
        X = X.view(X.shape[0], -1)  # flatten
        # F5
        X = self.fc5(X)
        X = F.relu(X)
        # F6
        X = self.fc6(X)
        X = F.relu(X)
        # OUTPUT
        X = self.out(X)
        X = F.softmax(X, dim=1)
        return X
    def test(model,testloader,name=""):
        model.eval()
        for x,y in testloader:
            y_hat = model(x)
            y_hat = y_hat.argmax(dim=1).type_as(y)
            y = y.argmax(dim=1)
        return y_hat, y