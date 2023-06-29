# 在该文件NeuralNetwork类中定义你的模型
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader


class ResBlock(torch.nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1)
        self.bn1 = torch.nn.BatchNorm2d(ch_out)
        self.conv2 = torch.nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2 = torch.nn.BatchNorm2d(ch_out)

        self.extra = torch.nn.Sequential()
        if ch_in != ch_out:
            # [b,ch_in,h,w] => [b,ch_out,h,w]
            self.extra = torch.nn.Sequential(
                torch.nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride),
                torch.nn.BatchNorm2d(ch_out)
            )
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        # short cut
        # extra module: [b,ch_in,h,w] => [b,ch_out,h,w]
        y = self.extra(x) + y
        return y

class ResNet18(torch.nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3,64,kernel_size=3,stride=3),
            torch.nn.BatchNorm2d(64)
        )
        # 接4个blocks
        # [b,64,h,w] => [b,128,h,w]
        self.blk1 = ResBlock(64,128,stride=2)
        # [b,128,h,w] => [b,256,h,w]
        self.blk2 = ResBlock(128,256,stride=2)
        # [b,256,h,w] => [b,512,h,w]
        self.blk3 = ResBlock(256,512,stride=2)
        # [b,512,h,w] => [b,1024,h,w]
        self.blk4 = ResBlock(512,512,stride=2)

        self.outlayer = torch.nn.Linear(512,10)
    def forward(self,x):
        x = F.relu(self.conv(x))
        # [b,64,h,w] => [b,512,h,w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # [b,512,h,w] => [b,512,1,1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0),-1)
        x = self.outlayer(x)
        return x


def read_data():
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True,
                                                 transform=torchvision.transforms.ToTensor())
    dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False,
                                               transform=torchvision.transforms.ToTensor())
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val


def main():
    model = NeuralNetwork()  # 若有参数则传入参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model.load_state_dict(torch.load(parent_dir + '/pth/model.pth'))
    return model
