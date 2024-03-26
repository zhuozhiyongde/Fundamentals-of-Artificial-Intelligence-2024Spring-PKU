# 深度学习第一课课上配套代码
# 导入Pytorch库
import torch
from torch.nn import Linear, ReLU, Softmax
from torch.nn import Sequential

torch.manual_seed(99999)

# Single neuron
# 单个神经元

# matrix multiplication 矩阵乘法
x = torch.tensor([[1., 2., 3.]]) # 1x3
w = torch.tensor([[-0.5], [0.2], [0.1]]) # 3x1

print("X:\n", x, "\nShape:", x.shape)
print("W:\n", w, "\nShape", w.shape)

# Bias 偏置
b1 = torch.tensor(0.5)

# 计算有偏置的矩阵乘法
z1 = torch.matmul(x, w)+b1

print("Z:\n", z1, "\nShape", z1.shape)

# Two outputs 两个输出

x = torch.tensor([[1., 2., 3.]]) # 1x3

w = torch.tensor([[-0.5, -0.3],
                  [0.2, 0.4],
                  [0.1, 0.15]]) # 3x2

print("X:\n", x, "\nShape:", x.shape)
print("W:\n", w, "\nShape", w.shape)

# Bias 偏置
b2 = torch.tensor([0.5, 0.4])

z2 = torch.matmul(x, w)+b2

print("Z:\n", z2, "\nShape", z2.shape)

# Activation function 激活函数
# Reference: https://en.wikipedia.org/wiki/Activation_function
# Sigmoid function
sigmoid_torch = torch.nn.Sigmoid()
a1 = sigmoid_torch(z1)
print("Result torch sigmoid:", a1)

# define your own activation function
class ActSigmoid(torch.nn.Module):
    def __init__(self):
        super(ActSigmoid, self).__init__()
        
    def forward(self, x):
        return 1/(1+torch.exp(-x))

sigmoid_act = ActSigmoid()
a1_m = sigmoid_act(z1)
print("Result your own sigmoid:", a1_m)

# Softmax
softmax_torch = torch.nn.Softmax(dim=1)
a2 = softmax_torch(z2)
print("Result torch softmax:", a2)

# define your own activation function
class ActSoftmax(torch.nn.Module):
    def __init__(self):
        super(ActSoftmax, self).__init__()
        
    def forward(self, x):
        return torch.exp(x)/torch.sum(torch.exp(x), dim=1, keepdim=True)

softmax_act = ActSoftmax()
a2_m = softmax_act(z2)
print("Result your own softmax:", a2_m)

# MLP 多层感知机
# 构建序列式模型
layer_list = []
layer_list.append(Linear(in_features=3, out_features=3, bias=True))
layer_list.append(ReLU())
layer_list.append(Linear(in_features=3, out_features=3, bias=True))
layer_list.append(ReLU())
layer_list.append(Linear(in_features=3, out_features=3, bias=True))
layer_list.append(ReLU())
layer_list.append(Linear(in_features=3, out_features=3, bias=True))
layer_list.append(Softmax(dim=1))

mlp = Sequential(*layer_list)

x_input = torch.tensor([[100., 200., 300.]]) # 1x3
output = mlp(x_input)# 前向传播
print("Network structure:")
print(mlp)
print("Neural network output: ", output.shape)

# Loss functions 损失函数
# 多分类交叉熵
target = torch.tensor([[0, 1, 0]],dtype=torch.float) # 1x3
print("Target value:{}".format(target.numpy()))
print("Neural network output:{}".format(output.detach().numpy()))
loss_fn = torch.nn.CrossEntropyLoss()
l_ce = loss_fn(output, target)
print("Loss cross entropy:", l_ce)

# Mean absolute error 平均绝对误差
def mae(output, target):
    return torch.mean(torch.abs(output - target))

# Mean squared error 平均平方误差
def mse(output, target):
    return torch.mean((output - target)**2)

y1 = torch.tensor([[1., 3., 5., 7.]])
y2 = torch.tensor([[2., 4., 6., 8.]])

# 计算MAE和MSE
l_mae = mae(y1, y2)
l_mse = mse(y1, y2)

print("Loss MAE: {} \nLoss MSE: {}".format(l_mae, l_mse))

mse_torch = torch.nn.MSELoss()
l_mse_torch = mse_torch(y1, y2)
print("Loss MSE torch: {}".format(l_mse_torch))



# 前向传播，计算损失，反向传播，梯度下降，更新参数
x_in = torch.tensor([[100., 200., 300.]])
target = torch.tensor([[1, 0, 0]],dtype=torch.float)
loss_fn = torch.nn.CrossEntropyLoss()

output = mlp(x_in)
l_ce = loss_fn(output, target)
l_ce.backward()
print("Gradient of layer 3's weights:\n{}".format(mlp[-2].weight.grad))

opt = torch.optim.SGD(mlp.parameters(), lr=0.1)
print("Before training, network layer 3's weights is: {}".format(mlp[-2].weight))

mlp.train()
for i in range(100):
    output = mlp(x_in)
    l_ce = loss_fn(output, target)

    l_ce.backward()
    opt.step()
    opt.zero_grad()

print("After training, network layer 3's weights is: {}".format(mlp[-2].weight))

# params_after = get_params(mlp)