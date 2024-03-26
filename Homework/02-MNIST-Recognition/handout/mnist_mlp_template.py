# 第一课作业
# 使用Pytorch训练MNIST数据集的MLP模型
# 1. 运行、阅读并理解mnist_mlp_template.py,修改网络结构和参数，增加隐藏层，观察训练效果
# 2. 使用Adam等不同优化器，添加Dropout层，观察训练效果
# 要求：10个epoch后测试集准确率达到98%以上

# 导入相关的包
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np

from matplotlib import pyplot as plt

# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy')
y_train = np.load('./mnist/y_train.npy')
X_val = np.load('./mnist/X_val.npy')
y_val = np.load('./mnist/y_val.npy')
X_test = np.load('./mnist/X_test.npy')
y_test = np.load('./mnist/y_test.npy')

# 定义MNIST数据集类
class MNISTDataset(Dataset):#继承Dataset类

    def __init__(self, data=X_train, label=y_train):
        '''
        Args:
            data: numpy array, shape=(N, 784)
            label: numpy array, shape=(N, 10)
        '''
        self.data = data
        self.label = label

    def __getitem__(self, index):
        '''
        根据索引获取数据,返回数据和标签,一个tuple
        '''
        data = self.data[index].astype('float32') #转换数据类型, 神经网络一般使用float32作为输入的数据类型
        label = self.label[index].astype('int64') #转换数据类型, 分类任务神经网络一般使用int64作为标签的数据类型
        return data, label

    def __len__(self):
        '''
        返回数据集的样本数量
        '''
        return len(self.data)

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 10)


    def forward(self, x):
        x = x.view(-1, 784)
        
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

# 实例化模型
model = Net()
model.to(device='cuda')
# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 定义数据加载器
train_loader = DataLoader(MNISTDataset(X_train, y_train), \
                            batch_size=64, shuffle=True)
val_loader = DataLoader(MNISTDataset(X_val, y_val), \
                            batch_size=64, shuffle=True)
test_loader = DataLoader(MNISTDataset(X_test, y_test), \
                            batch_size=64, shuffle=True)

# 定义训练参数
EPOCHS = 10

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
# 训练模型
for epoch in range(EPOCHS):
    # 训练模式
    model.train()

    loss_train = []
    acc_train = []
    correct_train = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device='cuda'), target.to(device='cuda')
        # 梯度清零
        optimizer.zero_grad()
        # 前向计算
        output = model(data)
        # 计算损失
        loss = criterion(output, target)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        # 打印训练信息
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        loss_train.append(loss.item())
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        correct = pred.eq(target.view_as(pred)).sum().item()
        correct_train += correct
        acc_train.append(100.* correct / len(data))

    history['train_loss'].append(np.mean(loss_train))
    history['train_acc'].append(np.mean(acc_train))
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        np.mean(loss_train), correct_train, len(train_loader.dataset),
        100. * correct_train / len(train_loader.dataset)))
    

    # 测试模式
    model.eval()
    val_loss = []
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device='cuda'), target.to(device='cuda')
            
            output = model(data)
            val_loss.append(criterion(output, target).item()) # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss = np.mean(val_loss)

    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    history['val_loss'].append(val_loss)
    history['val_acc'].append(100. *correct / len(val_loader.dataset))

# 画图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['val_acc'], label='val_acc')
plt.legend()
plt.show()

