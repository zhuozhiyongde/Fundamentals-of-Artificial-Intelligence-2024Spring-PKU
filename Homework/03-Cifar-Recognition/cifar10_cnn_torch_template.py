# 第二课作业
# 用pytorch实现卷积神经网络，对cifar10数据集进行分类
# 要求:1. 使用pytorch的nn.Module和Conv2d等相关的API实现卷积神经网络
#      2. 使用pytorch的DataLoader和Dataset等相关的API实现数据集的加载
#      3. 修改网络结构和参数，观察训练效果
#      4. 使用数据增强，提高模型的泛化能力

import os
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


# 定义超参数
batch_size = 0
learning_rate = 0
num_epochs = 0

# 定义数据预处理方式
# 普通的数据预处理方式
transform = transforms.Compose([
    transforms.ToTensor(),])
# 数据增强的数据预处理方式
# transform = transforms.Compose(



# 定义数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class Net(nn.Module):
    '''
    定义卷积神经网络,3个卷积层,2个全连接层
    '''
    pass

# 实例化模型
model = Net()

use_mlu = False
try:
    use_mlu = torch.mlu.is_available()
except:
    use_mlu = False

if use_mlu:
    device = torch.device('mlu:0')
else:
    print("MLU is not available, use GPU/CPU instead.")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

model = model.to(device)

# 定义损失函数和优化器
criterion = None
optimizer = None

# 训练模型
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (outputs.argmax(1) == labels).float().mean()

        # 打印训练信息
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), accuracy.item() * 100))

    # 测试模式
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))