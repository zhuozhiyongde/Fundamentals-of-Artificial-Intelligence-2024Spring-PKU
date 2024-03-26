# -*- coding: utf-8 -*-
"""
@ author: Yiliang Liu
"""


# 作业内容：更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集

import numpy as np

from tqdm  import tqdm


# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy') # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy') # (60000, )
y_train = np.eye(10)[y_train] # (60000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy') # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy') # (10000,)
y_val = np.eye(10)[y_val] # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy') # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy') # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码


# 定义激活函数
def relu(x):
    '''
    relu函数
    '''
    pass

def relu_prime(x):
    '''
    relu函数的导数
    '''
    pass

#输出层激活函数
def f(x):
    '''
    softmax函数, 防止除0
    '''
    pass

def f_prime(x):
    '''
    softmax函数的导数
    '''
    pass

# 定义损失函数
def loss_fn(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    pass

def loss_fn_prime(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    pass



# 定义权重初始化函数
def init_weights(shape=()):
    '''
    初始化权重
    '''
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0/shape[0]), size=shape)

# 定义网络结构
class Network(object):
    '''
    MNIST数据集分类网络
    '''

    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        '''
        初始化网络结构
        '''
        pass

    def forward(self, x):
        '''
        前向传播
        '''
        pass

    def step(self, x_batch, y_batch):
        '''
        一步训练
        '''

        # 前向传播
        pass
        # 计算损失和准确率
        pass
        
        # 反向传播
        pass


        # 更新权重
        pass


if __name__ == '__main__':
    # 训练网络
    net = Network(input_size=784, hidden_size=256, output_size=10, lr=0.01)
    for epoch in range(10):
        losses = []
        accuracies = []
        p_bar = tqdm(range(0, len(X_train), 64))
        for i in p_bar:
            pass
        