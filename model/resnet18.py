import torch.nn as nn
from torch.nn import functional as F

class resblock(nn.Module): # channel_in:输入通道 channel_out:输出通道
    def __init__(self,channel_in,channel_out,stride=1):
        # left
        super(resblock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channel_in,out_channels=channel_out,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn_1 = nn.BatchNorm2d(channel_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=channel_out,out_channels=channel_out,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn_2 = nn.BatchNorm2d(channel_out)
        # shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or channel_in != channel_out:
            self.shortcut = nn.Sequential(
              nn.Conv2d(in_channels=channel_in,out_channels=channel_out,kernel_size=1,stride=stride,bias=False) ,
              nn.BatchNorm2d(channel_out)
            )
        #
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self,x):
        # left
        x1 = self.conv1(x)
        x2 = self.bn_1(x1)
        x3 = self.relu(x2)
        x4 = self.conv2(x3)
        x5 = self.bn_2(x4)
        x6 = self.shortcut(x)
        out = x5+x6
        out = self.relu2(out)
        return out

class resnet18(nn.Module):
    def __init__(self,block):
        super(resnet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(block, 64, 2, stride=1)
        self.layer2 = self.make_layer(block, 128, 2, stride=2)
        self.layer3 = self.make_layer(block, 256, 2, stride=2)
        self.layer4 = self.make_layer(block, 512, 2, stride=2)
        self.fc = nn.Linear(512, 10)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        out6 = F.avg_pool2d(out5, 4)
        out7 = out6.view(out6.size(0), -1)
        out = self.fc(out7)
        return out,out5,out2

def Resnet18():
    return resnet18(resblock)

if __name__ == "__main__":

    net = Resnet18()
    net.eval()
    print(net)
    with open("F:/pycharm_project/20220315/log/netstructure/resnet18_structure.txt", "w") as f2:
        for param in net.named_parameters():
            print(param[0])
            f2.write(param[0])
            f2.write('\n')
            f2.flush()

