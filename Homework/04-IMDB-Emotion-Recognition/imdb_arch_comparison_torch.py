#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   imdb_arch_comparison_torch.py
# @Time    :   2024/04/28 00:10:24
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code


import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import LSTM, Embedding, Linear, Module
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from utils import Accuracy, load_imdb_dataset

# 初始化训练设备
print("use GPU/CPU.")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
num_of_layers = 1
batch_size = 64
hidden_size = 64

X_train, y_train, X_test, y_test = load_imdb_dataset(
    "data", nb_words=20000, test_split=0.2
)

seq_Len = 200
vocab_size = len(X_train) + 1


# 定义训练集
class ImdbDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        data = self.X[index]
        data = np.concatenate([data[:seq_Len], [0] * (seq_Len - len(data))]).astype(
            "int32"
        )  # set

        label = self.y[index]
        return data, label

    def __len__(self):
        return len(self.y)


# 定义网络结构
class ImdbNetLSTM(Module):
    def __init__(self):
        super(ImdbNetLSTM, self).__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=64)
        self.lstm = LSTM(input_size=64, hidden_size=64)
        self.linear1 = Linear(in_features=64, out_features=64)
        self.dropout = torch.nn.Dropout(0.5)
        self.linear2 = Linear(in_features=64, out_features=2)

    def forward(self, x):
        batch_size_in_forward = x.shape[0]
        prev_h = torch.zeros(num_of_layers, batch_size_in_forward, hidden_size).to(
            device
        )  # (num_layers, batch_size, hidden_size)
        prev_c = torch.zeros(num_of_layers, batch_size_in_forward, hidden_size).to(
            device
        )  # (num_layers, batch_size, hidden_size)
        x = self.embedding(x)
        x = x.permute(
            1, 0, 2
        )  # x经过permute将变成 (seq_len, batch_size, input_size), 便于适应LSTM输入
        x, _ = self.lstm(x, [prev_h, prev_c])
        x = torch.mean(
            x, dim=0
        )  # 对seq_len维度求均值，得到一个batch_size个、长为hidden_size的向量作为输出
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# 定义网络结构
class ImdbNetRNN(Module):
    def __init__(self):
        super(ImdbNetRNN, self).__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=64)
        self.rnn = torch.nn.RNN(input_size=64, hidden_size=64)
        self.linear1 = Linear(in_features=64, out_features=64)
        self.dropout = torch.nn.Dropout(0.5)
        self.linear2 = Linear(in_features=64, out_features=2)

    def forward(self, x):
        batch_size_in_forward = x.shape[0]
        prev_h = torch.zeros(num_of_layers, batch_size_in_forward, hidden_size).to(
            device
        )  # (num_layers, batch_size, hidden_size)
        x = self.embedding(x)
        x = x.permute(
            1, 0, 2
        )  # x经过permute将变成 (seq_len, batch_size, input_size), 便于适应LSTM输入
        x, _ = self.rnn(x, prev_h)
        x = torch.mean(
            x, dim=0
        )  # 对seq_len维度求均值，得到一个batch_size个、长为hidden_size的向量作为输出
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# 定义网络结构
class ImdbNetGRU(Module):
    def __init__(self):
        super(ImdbNetGRU, self).__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=64)
        self.gru = torch.nn.GRU(input_size=64, hidden_size=64)
        self.linear1 = Linear(in_features=64, out_features=64)
        self.dropout = torch.nn.Dropout(0.5)
        self.linear2 = Linear(in_features=64, out_features=2)

    def forward(self, x):
        batch_size_in_forward = x.shape[0]
        prev_h = torch.zeros(num_of_layers, batch_size_in_forward, hidden_size).to(
            device
        )  # (num_layers, batch_size, hidden_size)
        x = self.embedding(x)
        x = x.permute(
            1, 0, 2
        )  # x经过permute将变成 (seq_len, batch_size, input_size), 便于适应LSTM输入
        x, _ = self.gru(x, prev_h)
        x = torch.mean(
            x, dim=0
        )  # 对seq_len维度求均值，得到一个batch_size个、长为hidden_size的向量作为输出
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# 定义超参
n_epoch = 10
print_freq = 2

# 实例化训练集和网络
train_dataset = ImdbDataset(X=X_train, y=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = ImdbDataset(X=X_test, y=y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

netLSTM = ImdbNetLSTM()
netRNN = ImdbNetRNN()
netGRU = ImdbNetGRU()

writer = SummaryWriter(log_dir="./runs/rnn_vs_lstm_vs_gru")


# 分类训练
def train(model, device, train_loader, optimizer, epoch, arch):
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
    print(
        "Train Epoch: {} Loss: {:.6f} \t Acc: {:.6f}".format(
            epoch, train_loss / n_iter, train_acc / n_iter
        )
    )
    writer.add_scalars("Loss", {f"Train/{arch}": train_loss / n_iter}, epoch)
    writer.add_scalars("Accuracy", {f"Train/{arch}": train_acc / n_iter}, epoch)


def test(model, device, test_loader, arch):
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
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {:.4f}".format(test_loss, test_acc)
    )
    writer.add_scalars("Loss", {f"Test/{arch}": test_loss}, epoch)
    writer.add_scalars("Accuracy", {f"Test/{arch}": test_acc}, epoch)


metric = Accuracy()
optimizer = torch.optim.Adam(netLSTM.parameters(), lr=1e-3, weight_decay=1e-4)
print("------------------------")
print("vocab_size: ", vocab_size)
print(netLSTM)
gamma = 0.7
for epoch in range(1, n_epoch + 1):
    train(netLSTM, device, train_loader, optimizer, epoch, "LSTM")
    test(netLSTM, device, test_loader, "LSTM")

metric = Accuracy()
optimizer = torch.optim.Adam(netRNN.parameters(), lr=1e-3, weight_decay=1e-4)
print("------------------------")
print("vocab_size: ", vocab_size)
print(netRNN)
gamma = 0.7
for epoch in range(1, n_epoch + 1):
    train(netRNN, device, train_loader, optimizer, epoch, "RNN")
    test(netRNN, device, test_loader, "RNN")

metric = Accuracy()
optimizer = torch.optim.Adam(netGRU.parameters(), lr=1e-3, weight_decay=1e-4)
print("------------------------")
print("vocab_size: ", vocab_size)
print(netGRU)
gamma = 0.7
for epoch in range(1, n_epoch + 1):
    train(netGRU, device, train_loader, optimizer, epoch, "GRU")
    test(netGRU, device, test_loader, "GRU")

writer.close()
