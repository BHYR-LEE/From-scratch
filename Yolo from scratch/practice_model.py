import torch
import torch.nn as nn

architecture_config = [     ## kernel_size,output_channels,stride,padding
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

class Yolo2(nn.Module):
    def __init__(self,in_channels=3,**kwargs):
        super(Yolo2,self).__init__()
        self.in_channels = in_channels
        self.cnnpart = self._create_cnnpart()
        self.fcspart = self._create_fcspart(**kwargs)
    
    def forward(self,x):

        x = self.cnnpart(x)
        x = self.fcspart(torch.flatten(x,start_dim=1))
    
        return x

    def _create_cnnpart(self):
        in_channels = self.in_channels
        layers = []
        for x in architecture_config:
            if type(x) == tuple:
                layers += [CnnBlock(in_channels,x[1],x[0],x[2],x[3])]
                in_channels = x[1]
            if type(x) == str:
                layers += [nn.MaxPool2d(2,2)]
            if type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [CnnBlock(in_channels,conv1[1],conv1[0],conv1[2],conv1[3])]
                    layers += [CnnBlock(conv1[1],conv2[1],conv2[0],conv2[2],conv2[3])]
        return nn.Sequential(*layers)

    def _create_fcspart(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Linear(S * S * 1024, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5))   ## 나중에 reshape 해줄거임
            )


        

class CnnBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(CnnBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        return self.relu(self.batchnorm(self.conv(x)))



model = Yolo2(split_size=7,num_boxes=2,num_classes=20)
x = torch.randn((2,3,448,448))
print(model(x).shape)