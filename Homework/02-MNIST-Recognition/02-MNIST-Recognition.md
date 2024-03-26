# Homework 02 MNIST-Recognition

<center>
  北京大学 2024 春季人工智能基础第一次课程作业
</center>

---

<center>
  Arthals 2110306206
  <br/>
  <pre>zhuozhiyongde@126.com</pre>
  2024.03
</center>

---

## NumPy version

### 1. MLP with SGD

![np_mnist_mlp](README.assets/np_mnist_mlp.png)

epoch: 10, val_loss: 0.1605, val_acc: 0.9543

### 2. MLP with SGD and Momentum

![np_mnist_mlp_monentum](README.assets/np_mnist_mlp_monentum.png)

epoch: 10, val_loss: 0.0923, val_acc: 0.9723

## PyTorch version

### MLP with Adam and Dropout

![mlp_adam_dropout](README.assets/mlp_adam_dropout.png)

-   Training Epoch:10
-   Loss: 0.004142
-   Val acc: 97.94%

![loss_acc_cnn](README.assets/loss_acc_mlp.png)

### CNN

![cnn](README.assets/cnn.png)

-   Training Epoch:10
-   Loss: 0.001387
-   Val acc: 99.08%

![loss_acc_cnn](README.assets/loss_acc_cnn.png)
