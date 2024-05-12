#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   lstm_manual.py
# @Time    :   2024/04/28 00:11:19
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code


# 用Pytorch手写一个LSTM网络，在IMDB数据集上进行训练

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import Accuracy, load_imdb_dataset

print("using GPU/CPU.")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

X_train, y_train, X_test, y_test = load_imdb_dataset(
    "data", nb_words=20000, test_split=0.2
)

seq_Len = 200
vocab_size = len(X_train) + 1
print("vocab_size: ", vocab_size)


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


# 你需要实现的手写LSTM内容，包括LSTM类所属的__init__函数和forward函数
class LSTM(nn.Module):
    """
    手写 LSTM 网络
    输入数据 x: (batch_size, seq_len, input_size)
    输出数据/隐状态 h_n: (batch_size, hidden_size)
    输出数据/细胞状态 c_n: (batch_size, hidden_size)
    """

    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        # LSTM层
        self.input_size = input_size
        self.hidden_size = hidden_size

        # trick: 合并了矩阵乘法，以提高计算效率
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()

    def init_weights(self):
        # 初始化三个矩阵为 -sqrt(1/hidden_size) 到 sqrt(1/hidden_size) 之间的均匀分布
        boundary = 1 / np.sqrt(self.hidden_size)
        nn.init.uniform_(self.W, -boundary, boundary)
        nn.init.uniform_(self.U, -boundary, boundary)
        nn.init.uniform_(self.bias, -boundary, boundary)

    def forward(self, x):
        # 初始化隐状态和细胞状态
        batch_size, seq_len, _ = x.shape
        h_t, c_t = (
            torch.zeros(batch_size, self.hidden_size).to(x.device),
            torch.zeros(batch_size, self.hidden_size).to(x.device),
        )

        h_sum = torch.zeros(batch_size, self.hidden_size).to(x.device)

        # 遍历每个时间步
        for i in range(seq_len):
            x_t = x[:, i, :]
            all_gates = x_t @ self.W + h_t @ self.U + self.bias
            # 拆分出四个门的结果
            f_t = F.sigmoid(all_gates[:, : self.hidden_size])
            i_t = F.sigmoid(all_gates[:, self.hidden_size : 2 * self.hidden_size])
            g_t = F.tanh(all_gates[:, 2 * self.hidden_size : 3 * self.hidden_size])
            o_t = F.sigmoid(all_gates[:, 3 * self.hidden_size :])
            # 更新细胞状态和隐状态
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * F.tanh(c_t)
            h_sum += h_t

        return h_sum / seq_len


# 你需要实现网络推理和训练内容，仅需要完善forward函数
class Net(nn.Module):
    """
    一层LSTM的文本分类模型
    """

    def __init__(self, embedding_size=64, hidden_size=64, num_classes=2):
        super(Net, self).__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # LSTM层
        self.lstm = LSTM(input_size=embedding_size, hidden_size=hidden_size)
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: 输入, shape: (batch_size, seq_len)
        """

        # 词嵌入
        x = self.embedding(x)
        # LSTM层
        x = self.lstm(x)
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # 如果启用负对数似然 NLLLoss，则下面这行可以让损失恢复正常，因为这样组合后等价于交叉熵损失
        # x = F.log_softmax(x, dim=1)
        return x


n_epoch = 10
batch_size = 128

train_dataset = ImdbDataset(X=X_train, y=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ImdbDataset(X=X_test, y=y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

net = Net()
metric = Accuracy()
print(net)
writer = SummaryWriter(log_dir="./runs/lstm_manual")


def train(model, device, train_loader, optimizer, epoch):
    model = model.to(device)
    model.train()
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    train_acc = 0
    train_loss = 0
    n_iter = 0
    for batch_idx, (data, target) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
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
    writer.add_scalars("Loss", {"Train": train_loss / n_iter}, epoch)
    writer.add_scalars("Accuracy", {"Train": train_acc / n_iter}, epoch)


def test(model, device, test_loader, epoch):
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
    writer.add_scalars("Loss", {"Test": test_loss}, epoch)
    writer.add_scalars("Accuracy", {"Test": test_acc}, epoch)


optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)
gamma = 0.7
for epoch in range(1, n_epoch + 1):
    train(net, device, train_loader, optimizer, epoch)
    test(net, device, test_loader, epoch)

writer.close()
