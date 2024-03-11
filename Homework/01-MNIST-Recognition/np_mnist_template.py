#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author  :   Arthals
# @File    :   np_mnist_template.py
# @Time    :   2024/03/10 04:26:45
# @Contact :   zhuozhiyongde@126.com
# @Software:   Visual Studio Code


# 作业内容：更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集

import numpy as np

from tqdm import tqdm


# 加载数据集,numpy格式
X_train = np.load("./mnist/X_train.npy")  # (60000, 784), 数值在0.0~1.0之间
y_train = np.load("./mnist/y_train.npy")  # (60000, )
y_train = np.eye(10)[y_train]  # (60000, 10), one-hot编码

X_val = np.load("./mnist/X_val.npy")  # (10000, 784), 数值在0.0~1.0之间
y_val = np.load("./mnist/y_val.npy")  # (10000,)
y_val = np.eye(10)[y_val]  # (10000, 10), one-hot编码

X_test = np.load("./mnist/X_test.npy")  # (10000, 784), 数值在0.0~1.0之间
y_test = np.load("./mnist/y_test.npy")  # (10000,)
y_test = np.eye(10)[y_test]  # (10000, 10), one-hot编码


# 定义激活函数
def relu(x):
    """
    relu函数
    """
    return np.maximum(0, x)


def relu_prime(x):
    """
    relu函数的导数
    """
    return np.where(x > 0, 1, 0)


# 输出层激活函数
def softmax(x: np.ndarray) -> np.ndarray:
    """
    softmax函数, 防止除0
    x: (batch_size, dimensions)
    """
    stable_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return stable_x / np.sum(stable_x, axis=1, keepdims=True)


def softmax_prime(x):
    """
    softmax函数的导数
    """
    return softmax(x) * (1.0 - softmax(x))


# 定义损失函数
def loss_fn(y_true, y_pred):
    """
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    """
    return -np.sum(y_true * np.log(y_pred + 1e-8), axis=-1)


def loss_fn_prime(y_true, y_pred):
    """
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    """
    return y_pred - y_true


# 定义权重初始化函数
def init_weights(shape=()):
    """
    初始化权重
    """
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0 / shape[0]), size=shape)


# 定义网络结构
class Network(object):
    """
    MNIST数据集分类网络
    """

    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        """
        初始化网络结构
        """
        self.lr = lr
        self.W1 = init_weights((input_size + 1, hidden_size))
        self.W2 = init_weights((hidden_size, hidden_size))
        self.W3 = init_weights((hidden_size, output_size))

    def forward(self, x):
        """
        前向传播
        """
        # 将 x 添加偏置
        x = np.hstack([x, np.ones((x.shape[0], 1))])
        self.z1 = x @ self.W1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3
        self.a3 = softmax(self.z3)
        return self.a3

    def step(self, x_batch, y_batch):
        """
        一步训练
        """
        batch_size = x_batch.shape[0]

        # 前向传播
        y_pred = self.forward(x_batch)

        # 计算损失和准确率
        loss = np.mean(loss_fn(y_batch, y_pred))
        acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))

        # 反向传播
        grad_a3 = loss_fn_prime(y_batch, y_pred)
        grad_W3 = self.a2.T @ grad_a3 / batch_size

        grad_a2 = grad_a3 @ self.W3.T * relu_prime(self.z2)
        grad_W2 = self.a1.T @ grad_a2 / batch_size

        grad_a1 = grad_a2 @ self.W2.T * relu_prime(self.z1)
        x_prime = np.hstack([x_batch, np.ones((x_batch.shape[0], 1))])
        grad_W1 = x_prime.T @ grad_a1 / batch_size

        # 更新权重
        self.W3 -= self.lr * grad_W3
        self.W2 -= self.lr * grad_W2
        self.W1 -= self.lr * grad_W1

        return loss, acc


if __name__ == "__main__":
    # 训练网络
    net = Network(input_size=784, hidden_size=256, output_size=10, lr=0.01)
    for epoch in range(10):
        losses = []
        accuracies = []
        p_bar = tqdm(range(0, len(X_train), 64))
        for i in p_bar:
            x_batch = X_train[i : i + 64]
            y_batch = y_train[i : i + 64]
            loss, acc = net.step(x_batch, y_batch)
            losses.append(loss)
            accuracies.append(acc)

            p_bar.set_description(f"loss: {loss:.4f}, acc: {acc:.4f}")

        # 验证网络
        y_pred = net.forward(X_val)
        val_loss = np.mean(loss_fn(y_val, y_pred))
        val_acc = np.mean(np.argmax(y_pred, axis=-1) == np.argmax(y_val, axis=-1))
        print(f"epoch: {epoch}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
