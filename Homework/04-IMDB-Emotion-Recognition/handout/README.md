# 作业 4：imdb lstm 分类实验

注，所有代码在如下版本中测试通过

```
Python=3.9
Pytorch=1.13.1+cu117
numpy=1.26.4
```

理论上其他的 pytorch 版本也能通过

## 作业 ① LSTM RNN GRU 对比试验

### 基础代码

`example_imdb_lstm_torch.py`

在这个代码当中：

-   Line 14-30 初始化训练设备
-   Line 42-57 定义训练集
-   Line 60-77 定义网络结构
-   Line 80-89 定义超参，实例化训练集和网络

### 执行命令

```bash
python  example_imdb_lstm_torch.py
```

训练 5 epoch, 耗时约 1min，训练分类精度为 0.931

### LSTM 的参考结果

```log
vocab_size:  20001
ImdbNet(
  (embedding): Embedding(20001, 64)
  (lstm): LSTM(64, 64)
  (linear1): Linear(in_features=64, out_features=64, bias=True)
  (act1): ReLU()
  (linear2): Linear(in_features=64, out_features=2, bias=True)
)
Train Epoch: 1 Loss: 0.592848    Acc: 0.665735
Test set: Average loss: 0.4720, Accuracy: 0.7789
Train Epoch: 2 Loss: 0.390458    Acc: 0.827177
Test set: Average loss: 0.3778, Accuracy: 0.8319
Train Epoch: 3 Loss: 0.297707    Acc: 0.877496
Test set: Average loss: 0.3528, Accuracy: 0.8449
Train Epoch: 4 Loss: 0.237297    Acc: 0.908047
Test set: Average loss: 0.3485, Accuracy: 0.8558
Train Epoch: 5 Loss: 0.185850    Acc: 0.931410
Test set: Average loss: 0.3699, Accuracy: 0.8523
```

### 你的任务

阅读 Pytorch 内置的 [Recurrent Layers](https://pytorch.org/docs/stable/nn.html#recurrent-layers) 的官方文档（包括 LSTM，RNN，GRU），了解不同的 Recurrent Layers 的输入和输出结构，以及初始化参数的含义。请在实验报告当中任意挑选一种，简单介绍它的输入输出的格式、以及初始化参数的含义。（1 分）

修改 `ImdbNet` 中的 `self.lstm` 为上述三种内置的 Layer（LSTM，RNN，GRU；其中原始代码中已经填充了 LSTM），运行代码并在实验报告当中汇报结果，结果格式请参考上面的 “**LSTM 的参考结果**”（2 分）

## 作业 ② 手写 LSTM 实验

该作业参考时间：

GPU 1 分钟 1 个 epoch

CPU 5 分钟 1 个 epoch

### 基础代码

`lstm_manual_template.py`

-   Line 56-66 是你需要实现的手写 LSTM 内容，包括 LSTM 类所属的 `__init__` 函数和 `forward` 函数
-   Line 69-92 是你需要实现网络推理和训练内容，仅需要完善 `forward` 函数
-   Line 97-136 训练代码，需要修改超参以完成实验内容

### 你的任务

在不使用 `nn.LSTM` 的情况下，从原理上实现 LSTM。

你可以参考 PPT 或者 Pytorch 官方文档 [LSTM — PyTorch 2.2 documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM) 来完成这个任务。

训练后测试集准确率要求不低于 80% ，你需要在实验报告当中汇报结果，结果的格式请参考上面的 “**LSTM 的参考结果**”（2 分）

我们会检查代码实现的正确性（2 分）

调整网络结构（例如网络隐藏层维度，1 分）、损失函数（其它的损失函数可以参考 Pytorch 的官方文档 [torch.nn#loss-functions](https://pytorch.org/docs/stable/nn.html#loss-functions)，1 分）、训练流程（例如训练的超参数，epoch、batchsize 等，1 分），观察他们对训练效果的影响。

**注：80% 的准确率仅要求最优结果。在调整网络结构、损失函数、训练流程当中，不要求达到 80% 准确率。**

## 最后，你需要提交的内容

### 一份实验报告

内容包括

1、作业 ① 中，任选一种 Recurrent Layers 的简介（1 分）

2、作业 ① 中，LSTM RNN GRU 对比实验的实验结果（2 分）

3、作业 ② 中，超过 80% 实验结果的截图（2 分）

4、作业 ② 中，调整三个不同内容的结果截图（3 分）

### 代码文件

内容包括整个作业包，其中必须包括手写 LSTM 的代码（正确实现，2 分）。
