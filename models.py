## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # output layer dim = (W-F)/S + 1
        # output dim (224-5)/1 +1 = 220 => maxpool => 110
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.bn32 = nn.BatchNorm2d(32)
        
        # output dim (110-5)/1 +1 = 106 => maxpool => 52
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.bn64 = nn.BatchNorm2d(64)
        
        # output dim (52-5)/1 +1 = 48 => maxpool => 24
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.bn128 = nn.BatchNorm2d(128)
        
        # output dim (24-3)/1 +1 = 22 => maxpool => 11
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.bn256 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(11*11*256, 4096)
        self.bn_fc1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024,136)
        
        self.dropout = nn.Dropout(0.5)
        

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.leaky_relu(self.bn32(self.conv1(x))))
        x = self.pool(F.leaky_relu(self.bn64(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn128(self.conv3(x))))
        x = self.pool(F.leaky_relu(self.bn256(self.conv4(x))))
        
        x = x.view(x.size(0),-1)
        
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
