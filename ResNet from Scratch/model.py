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

        x = self.covn1(x)
        x = self.bh1(x)
        x = self.relu(x)

        x = self.covn2(x)
        x = self.bh2(x)
        x = self.relu(x)
        x = self.covn3(x)
        x = self.bh3(x)
        x = self.relu(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        out = self.relu(identity + x)

        return out



class ResNet(nn.Module): ## [3,4,6,3]
    def __init__(self, layers, image_channels, num_classes):
        super(ResNet,self).__init__()
        self.in_channels = 64 
        self.downsampling = nn.Conv2d(image_channels,in_channels,kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)  ## (N, 64, 56,56)
        self.layer1 = _make_layer(3,64,64)
        self.layer2 = _make_layer(4,256,128)
        self.layer3 = _make_layer(6,512,256)
        self.layer4 = _make_layer(4,1024,512)
        self.fc = nn.Linear(4096,1000)
        
        

    def _make_layer(self, num_residual_blocks,in_channels,out_channels):
        layers = []
        for i in range(num_residual_blocks):
            layers.append(Block
                        (
                            in_channels,
                            out_channels,
                            identity_downsample=nn.Conv2d(in_channels,out_channels,1,1,0)
                        )
            )
            in_channels = out_channels*4
        
        return nn.Sequential(*layers)







