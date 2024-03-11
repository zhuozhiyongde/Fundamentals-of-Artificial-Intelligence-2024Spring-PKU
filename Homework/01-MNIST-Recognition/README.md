# MNIST-Recognition

Softmax 导数证明

$$
\begin{align*}
\frac{\partial S(\mathbf{x})_j}{\partial x_i} &= \frac{\partial}{\partial x_i} \left(\frac{e^{x_j}}{\sum_{k=1}^n e^{x_k}}\right) \\
&= \frac{e^{x_j}\delta_{ij}\sum_{k=1}^n e^{x_k} - e^{x_j}e^{x_i}}{\left(\sum_{k=1}^n e^{x_k}\right)^2} \\
&= \delta_{ij} S(\mathbf{x})_j - S(\mathbf{x})_i S(\mathbf{x})_j
\end{align*}
$$
