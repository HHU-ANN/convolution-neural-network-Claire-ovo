# 在该文件NeuralNetwork类中定义你的模型
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms


# 定义神经网络
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(4096, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.fc3(x)
        return x
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# 加载CIFAR-10数据集
train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
# 创建模型实例
model = NeuralNetwork()
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# 训练模型
num_epochs = 10
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 打印训练信息
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
# 保存模型参数
torch.save(model.state_dict(), 'pth/model.pth')

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
