# -*- coding: utf-8 -*-

# 使用numpy实现反向传播算法示例
# 作业内容：1. 阅读并理解代码
#          2. 修改np_mnist_template.py，更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集

import numpy as np

X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]]) # 输入数据
y = np.array([[0], [1], [1], [1]]) # 标签数据

def sigmoid(x):
    '''
    sigmoid函数
    '''
    return 1. / (1. + np.exp(-x))

def sigmoid_prime(x):
    '''
    sigmoid函数的导数
    '''
    return sigmoid(x) * (1. - sigmoid(x))

def get_weights(shape=()):
    '''
    生成权重矩阵
    '''
    np.random.seed(seed=0)
    return np.random.normal(loc=0.0, scale=0.001, size=shape, )

class Network(object):
    '''
    定义网络结构
    '''
    def __init__(self, lr=0.01):
        '''
        初始化网络参数
        '''
        self.W1 = get_weights((3, 4)) # 输入层到隐藏层的权重矩阵
        self.b1 = get_weights((4)) # 输入层到隐藏层的偏置
        self.W2 = get_weights((4, 1)) # 隐藏层到输出层的权重矩阵
        self.b2 = get_weights((1)) # 隐藏层到输出层的偏置
        self.lr = lr # 学习率

    def backward(self, X_batch, y_batch):
        '''
        反向传播算法
        '''
        # 初始化梯度
        batch_size = 0
        batch_loss = 0
        batch_acc  = 0

        self.grads_W2 = np.zeros_like(self.W2)
        self.grads_b2 = np.zeros_like(self.b2)
        self.grads_W1 = np.zeros_like(self.W1)
        self.grads_b1 = np.zeros_like(self.b1)

        # 计算梯度
        for x, y in zip(X_batch, y_batch):
            ## forward
            z1  = np.matmul(x, self.W1) + self.b1 # z^{L-1}
            a1  = sigmoid(z1) # a^{L-1}
            z2  = np.matmul(a1, self.W2) + self.b2 # z^{L}
            a2 = sigmoid(z2) # a^{L}

            ## loss 不同的loss函数，梯度计算方式不同
            # 1 : mse (L2 loss)
            # loss = 1/2 * np.mean((y-a2)**2) # d_C/ d_a^{L} = (a^{L} - y)
            # delta_L = (a2 - y) * (a2 * (1-a2))

            # 2 : abs (L1 loss)
            # loss = np.mean(abs(y - a2))  # |x|' = (1 if x>0 else -1)
            # delta_L = (1. if (a2 - y > 0) else -1.) * a2 * (1-a2)

            # 3 : sigmoid cross entropy (logistic regression)
            loss = - (y*np.log(a2) + (1-y)*np.log(1-a2)) 
            delta_L = - (y * (1-a2) - (1-y) * a2 )
            #print(delta_L)
            
            # 梯度计算
            delta_l = np.matmul(self.W2, delta_L) * a1 * (1-a1)
            #print(delta_l)

            # 梯度累加
            self.grads_W2 += np.array([(a1 * delta_L)]).T
            #print(self.grads_W2)
            self.grads_b2 += delta_L
            #print(self.grads_b2)
            self.grads_W1 += np.matmul(np.array([x]).T, [delta_l])
            #print(self.grads_W1)
            self.grads_b1 += delta_l
            #print(self.grads_b1)

            # loss 累加
            batch_size += 1
            batch_loss += loss
            batch_acc += 1 if (y == (1 if a2>0.5 else 0)) else 0

        # 梯度平均
        self.grads_W2 /= batch_size
        self.grads_b2 /= batch_size
        self.grads_W1 /= batch_size
        self.grads_b1 /= batch_size
        # loss 平均
        batch_loss /= batch_size
        batch_acc /= batch_size

        print("loss:{} batch_acc:{} batch_size:{} lr:{}".format(batch_loss, batch_acc, batch_size, self.lr))

        # 更新参数
        self.W2 -= self.lr * self.grads_W2
        self.b2 -= self.lr * self.grads_b2
        self.W1 -= self.lr * self.grads_W1
        self.b1 -= self.lr * self.grads_b1

if __name__ == "__main__":
    net = Network(lr=10) # 构建网络，学习率为0.01

    # 在100个batch上训练,执行100次反向传播
    for i in range(100):
        net.backward(X, y)
