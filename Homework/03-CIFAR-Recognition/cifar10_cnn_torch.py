#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   cifar10_cnn_torch.py
# @Time    :   2024/03/31 01:17:11
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code

# Epoch [100/100]
# - Train Loss: 0.034, Train Accuracy: 98.93%
# - Test Loss: 0.5295, Test Accuracy: 90.06%

# 第二课作业
# 用pytorch实现卷积神经网络，对cifar10数据集进行分类
# 要求:1. 使用pytorch的nn.Module和Conv2d等相关的API实现卷积神经网络
#      2. 使用pytorch的DataLoader和Dataset等相关的API实现数据集的加载
#      3. 修改网络结构和参数，观察训练效果
#      4. 使用数据增强，提高模型的泛化能力

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

# 定义超参数
batch_size = 256
learning_rate = 0.001
num_epochs = 100

# 定义数据预处理方式
# 普通的数据预处理方式
# 数据预处理，包括数据增强
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# 加载数据集
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)

# 定义数据加载器
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=24
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=24
)


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # VGG16
        self.features = nn.Sequential(
            # From: 3x32x32
            # 1 block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # After: 64x16x16
            # 2 block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # After: 128x8x8
            # 3 block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # After: 256x4x4
            # 4 block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # After: 512x2x2
            # 5 block
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # After: 512x1x1
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# 实例化模型
model = Net()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

writer = SummaryWriter()

# 训练模型
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    running_loss = 0.0
    running_corrects = 0
    train_loader_bar = tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
    )
    for i, (images, labels) in enumerate(train_loader_bar):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)

    writer.add_scalars(
        "Loss", {"Train": running_loss / len(train_loader.dataset)}, epoch
    )
    writer.add_scalars(
        "Accuracy",
        {"Train": running_corrects.double() / len(train_loader.dataset)},
        epoch,
    )

    # 打印
    print(
        "Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%".format(
            epoch + 1,
            num_epochs,
            loss.item(),
            running_corrects.double() / len(train_loader.dataset) * 100.0,
        )
    )

    # 测试模式
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # 在每个 epoch 结束后记录总损失和准确率
        writer.add_scalars(
            "Loss", {"Test": running_loss / len(test_loader.dataset)}, epoch
        )
        writer.add_scalars(
            "Accuracy",
            {"Test": running_corrects.double() / len(test_loader.dataset)},
            epoch,
        )

    # 打印
    print(
        "Test Loss: {:.4f}, Test Accuracy: {:.2f}%".format(
            running_loss / len(test_loader.dataset) * 100.0,
            running_corrects.double() / len(test_loader.dataset) * 100.0,
        )
    )

# 关闭 SummaryWriter
writer.close()
