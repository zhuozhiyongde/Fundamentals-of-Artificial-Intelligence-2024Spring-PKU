# The same set of code can switch the backend with one line
import os

import torch
from torch.nn import Module
from torch.nn import Linear, LSTM, Embedding
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import sys
from utils import load_imdb_dataset, Accuracy
# 初始化训练设备
use_mlu = False
# try:
#     import torch_mlu
#     import torch_mlu.core.mlu_model as ct
#     global ct
#     use_mlu = torch.mlu.is_available()
# except:
#     use_mlu = False

if use_mlu:
    device = torch.device('mlu:0')
else:
    print("MLU is not available, use GPU/CPU instead.")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
num_of_layers = 1
batch_size = 64
hidden_size = 64

X_train, y_train, X_test, y_test = load_imdb_dataset('data', nb_words=20000, test_split=0.2)

seq_Len = 200
vocab_size = len(X_train) + 1
print("vocab_size: ", vocab_size)


# 定义训练集
class ImdbDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):

        data = self.X[index]
        data = np.concatenate([data[:seq_Len], [0] * (seq_Len - len(data))]).astype('int32')  # set

        label = self.y[index]
        return data, label

    def __len__(self):

        return len(self.y)


# 定义网络结构
class ImdbNet(Module):

    def __init__(self):
        super(ImdbNet, self).__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=64)
        self.lstm = LSTM(input_size=64, hidden_size=64)
        self.linear1 = Linear(in_features=64, out_features=64)
        self.act1 = torch.nn.ReLU()
        self.linear2 = Linear(in_features=64, out_features=2)

    def forward(self, x):
        batch_size_in_forward = x.shape[0]
        prev_h = torch.zeros(num_of_layers, batch_size_in_forward, hidden_size).to(device)  #(num_layers, batch_size, hidden_size)
        prev_c = torch.zeros(num_of_layers, batch_size_in_forward, hidden_size).to(device)  #(num_layers, batch_size, hidden_size)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # x经过permute将变成 (seq_len, batch_size, input_size), 便于适应LSTM输入
        x, _ = self.lstm(x, [prev_h, prev_c])
        x = torch.mean(x, dim=0)  # 对seq_len维度求均值，得到一个batch_size个、长为hidden_size的向量作为输出
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x


# 定义超参
n_epoch = 5
print_freq = 2

# 实例化训练集和网络
train_dataset = ImdbDataset(X=X_train, y=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = ImdbDataset(X=X_test, y=y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
net = ImdbNet()
metric = Accuracy()
print(net)


# 分类训练
def train(model, device, train_loader, optimizer, epoch):
    model = model.to(device)
    model.train()
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    train_acc = 0
    train_loss = 0
    n_iter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.long()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        metric.update(output, target)
        train_acc += metric.result()
        train_loss += loss.item()
        metric.reset()
        n_iter += 1
    print('Train Epoch: {} Loss: {:.6f} \t Acc: {:.6f}'.format(epoch, train_loss / n_iter, train_acc / n_iter))


def test(model, device, test_loader):
    model = model.to(device)
    model.eval()
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    test_loss = 0
    test_acc = 0
    n_iter = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += loss_func(output, target).item()
            metric.update(output, target)
            test_acc += metric.result()
            metric.reset()
            n_iter += 1
    test_loss /= n_iter
    test_acc /= n_iter
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))


optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0)
gamma = 0.7
for epoch in range(1, n_epoch + 1):
    train(net, device, train_loader, optimizer, epoch)
    test(net, device, test_loader)
