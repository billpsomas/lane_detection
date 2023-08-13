# refence https://github.com/SeungyounShin/LaneNet/blob/master/models/HNet.py
import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv =  conv3x3(inplanes, planes, stride)
        self.conv1 = conv3x3(planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.conv(x)
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out

class HNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = BasicBlock(3,16)
        self.block2 = BasicBlock(16,32)
        self.block3 = BasicBlock(32,64)
        self.pool = nn.MaxPool2d(2)

        self.head = nn.Sequential(nn.Linear(9216, 1024),
                                  nn.BatchNorm1d(1024),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(1024, 6),)

    def forward(self, x):
        batch = x.size(0)
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.view(batch,-1)
        #print(x.shape)
        x = self.head(x)

        return x


if __name__ == "__main__":
    import time

    hnet = HNet()
    x = torch.randn(2,3,128,72)

    startT = time.time()
    out = hnet(x)
    endT = time.time()


    print("forward time : ",endT - startT)
    print("output shape : ",out.shape)

    H = torch.zeros(2,3,3)
    H[:,0,0] = out[:,0] #a
    H[:,0,1] = out[:,1] #b
    H[:,0,2] = out[:,2] #c
    H[:,1,1] = out[:,3] #d
    H[:,1,2] = out[:,4] #e
    H[:,2,1] = out[:,5] #f
    H[:,-1,-1] = 1
    print(H)