import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block,self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bh1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bh2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, stride=1, padding=1)
        self.bh3 = nn.BatchNorm2d(self.expansion*out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
    
    def forward(self, x):
        identity = x 

        x = self.conv1(x)
        x = self.bh1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bh2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bh3(x)
        x = self.relu(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        out = self.relu(identity + x)

        return out

class ResNet(nn.Module): ## [3,4,6,3]
    def __init__(self, Block, layers, image_channels, num_classes):
        super(ResNet,self).__init__()
        self.in_channels = 64 

        self.downsampling = nn.Conv2d(image_channels,64 ,kernel_size=7, stride=2, padding=3)
        self.bh1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)  ## (N, 64, 56,56)
        self.layer1 = self._make_layer(Block, layers[0], 64, stride=1)
        self.layer2 = self._make_layer(Block, layers[1], 128, stride=2)
        self.layer3 = self._make_layer(Block, layers[2], 256, stride=2)
        self.layer4 = self._make_layer(Block, layers[3], 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.downsampling(x)
        x = self.bh1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x



    def _make_layer(self, Block, num_residual_blocks, out_channels, stride):
        identity_downsample =None
        layers = []

        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels*4))
        
        layers.append(Block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4

        for i in range(num_residual_blocks -1):
            layers.append(Block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)


net = ResNet(Block,[3,4,6,3],3,1000)

x = torch.randn(2,3,224,224)
y = net(x)
print(y.shape)