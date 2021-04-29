import torch
import torch.nn as nn

## architecture 
## kernel_size, out_channels, stride, padding
architecture_config = [
    (7,64,2,3),
    "M",
    (3,192,1,1),
    "M",
    (1,128,1,0),
    (3,256,1,1),
    (1,256,1,0),
    (3,512,1,1),
    "M",
    
    [(1,256,1,0),(3,512,1,1),4],    ## 4번 반복하는 부분
    
    (1,512,1,0),
    (3,1024,1,1),
    "M",

    [(1,512,1,0),(3,1024,1,1),2], ## 2번

    (3,1024,1,1),
    (3,1024,2,1),
    (3,1024,1,1),
    (3,1024,1,1),
]


class CNNBlock(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(CNNBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.conv(x)
        return self.leakyrelu(self.batchnorm(x))

        


class Yolo(nn.Module):
    def __init__(self, in_channels=3,**kwargs):
        super(Yolo,self).__init__()
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers()
        self.fcs = self._create_fcs(**kwargs)
    
    def forward(self,x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self): ## darknet
        layers = []
        in_channels = self.in_channels

        for x in architecture_config:
            if type(x) == tuple:
                layers += [
                    CNNBlock(
                    in_channels,x[1],kernel_size=x[0],padding=x[3],stride=x[2]
                    )
                ]
                in_channels = x[1]

            if type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            
            if type(x) == list:
                conv1 = x[0] ## tuple
                conv2 = x[1] ## tuple
                num_repeats = x[2] ## interger
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                        in_channels,conv1[1],kernel_size=conv1[0],padding=conv1[3],stride=conv1[2]
                        )                
                    ]
                    layers += [
                        CNNBlock(
                        conv1[1],conv2[1],kernel_size=conv2[0],padding=conv2[3],stride=conv2[2]                    
                        )
                        
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)
    
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Linear(S * S * 1024, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5))   ## 나중에 reshape 해줄거임
        )

