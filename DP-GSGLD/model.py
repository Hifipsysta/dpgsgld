


import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 100)
        )
        
    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out


class View(nn.Module):
    """
        Implements a reshaping module.
        Allows to reshape a tensor between NN layers.
    """

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)

class MNISTConvNet(nn.Module):

    def __init__(self, nChannels=1, ndf=64, filterSize=5, w_out=4, h_out=4, nClasses=10):
        super(MNISTConvNet, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(nChannels, ndf, filterSize),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(ndf),
            nn.Conv2d(ndf, ndf, filterSize),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2),
            View(-1, ndf * w_out * h_out),
            #PrintLayer("View"),
            #View(-1, 784),
            nn.Linear(ndf * w_out * h_out, 384),
            nn.SELU(inplace=True),
            nn.Linear(384, 192),
            nn.SELU(inplace=True),
            nn.Linear(192, nClasses),
            #nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class CIFARConvNet(nn.Module):

    def __init__(self, nChannels=3, ndf=64, filterSize=5, w_out=5, h_out=5, nClasses=10):
        super(CIFARConvNet, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(nChannels, ndf, filterSize),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(ndf),
            nn.Conv2d(ndf, ndf, filterSize),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            View(-1, ndf * w_out * h_out),
            #PrintLayer("View"),
            nn.Linear(ndf * w_out * h_out, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, nClasses),
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.classifier(x)
        return x


class LFWConvNet(nn.Module):

    def __init__(self, nClasses, ndf=64, w_out=5, h_out=5):
        super(LFWConvNet, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = ndf, kernel_size = 5, stride = 2, padding = 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 2,  stride = 2),
            #nn.BatchNorm2d(num_features = 6),
            nn.Conv2d(in_channels = 64, out_channels = ndf, kernel_size = 5, stride = 2, padding = 0),
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(num_features = 6),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.classifier = nn.Sequential(
            View(-1, ndf * w_out * h_out), 
            #PrintLayer("View"),
            nn.Linear(ndf * w_out * h_out, 384), 
            nn.ReLU(inplace=True),
            nn.Linear(384, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, nClasses),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        print('=======before full-connect=========', x.shape)
        x = self.classifier(x)
        print('========after full-connect=========', x.shape)
        return x


