<h1 align="center">CS231n 2025 学习笔记（四）：神经网络与反向传播</h1>

<p align="center">
  Neural Networks · Forward Propagation · Backpropagation · Jacobian · Matrix Calculus
</p>

> 本篇围绕神经网络的前向传播与反向传播展开，重点解释全连接层维度、内积、计算图、链式法则、雅可比矩阵和矩阵求导。内容以直观例子为主，适合作为 CS231n 第四讲的复习笔记。

## 本节重点

| 模块 | 核心问题 |
|---|---|
| 神经网络结构 | 多层线性变换为什么必须加入非线性激活函数？ |
| 前向传播 | 数据如何经过全连接层和激活函数得到预测结果？ |
| 反向传播 | 如何利用链式法则把梯度从损失传回各层？ |
| 矩阵求导 | 雅可比矩阵、转置和梯度形状应该如何理解？ |

<details>
<summary><strong>目录</strong></summary>

- [一、从线性分类器到神经网络](#section-01)
- [二、为什么必须加入激活函数](#section-02)
- [三、常见激活函数](#section-03)
- [四、全连接网络中的维度](#section-04)
- [五、输入维度如何确定](#section-05)
- [六、隐藏层神经元个数可以不同吗](#section-06)
- [七、内积与神经元计算](#section-07)
- [八、两层神经网络的前向传播](#section-08)
- [九、什么是计算图](#section-09)
- [十、反向传播的核心：链式法则](#section-10)
- [十一、常见节点的反向传播规律](#section-11)
- [十二、什么是雅可比矩阵](#section-12)
- [十三、雅可比矩阵例子](#section-13)
- [十四、线性层的雅可比矩阵](#section-14)
- [十五、ReLU 的雅可比矩阵](#section-15)
- [十六、矩阵乘法的反向传播](#section-16)
- [十七、两层神经网络的反向传播](#section-17)
- [十八、完整训练代码](#section-18)
- [十九、为什么不直接构造完整雅可比矩阵](#section-19)
- [二十、不同卷积滤波器之间有什么区别](#section-20)
- [二十一、一个滤波器如何计算](#section-21)
- [二十二、容易混淆的问题](#section-22)
- [二十三、本讲核心总结](#section-23)

</details>

本讲主要解决两个问题：

1. 如何用多层神经网络学习复杂的非线性关系？
2. 如何通过反向传播计算每个参数的梯度？

神经网络的训练流程是：

$$
\boxed{
\text{前向传播}
\rightarrow
\text{计算损失}
\rightarrow
\text{反向传播}
\rightarrow
\text{更新参数}
}
$$

<a id="section-01"></a>

## 一、从线性分类器到神经网络

线性分类器的得分函数为：

$$
S=XW+b
$$

它只能得到线性分类边界，难以处理复杂的非线性数据。

两层神经网络可以写成：

$$
H=f(XW_1+b_1)
$$

$$
S=HW_2+b_2
$$

合并后：

$$
\boxed{
S=f(XW_1+b_1)W_2+b_2
}
$$

其中：

- $X$：输入数据；
- $W_1,W_2$：权重矩阵；
- $b_1,b_2$：偏置；
- $H$：隐藏层特征；
- $f$：激活函数；
- $S$：最终类别得分。

这里的“两层”通常指两个包含可学习参数的线性层，输入层不计入层数。

<a id="section-02"></a>

## 二、为什么必须加入激活函数

假设两层之间没有激活函数：

$$
H=XW_1
$$

$$
S=HW_2=XW_1W_2
$$

令：

$$
W'=W_1W_2
$$

则：

$$
S=XW'
$$

它仍然只是一个线性模型。

因此，无论叠加多少个线性层，只要没有非线性激活函数，最终都可以合并成一个线性层。

$$
\boxed{
\text{激活函数负责为神经网络引入非线性}
}
$$

<a id="section-03"></a>

## 三、常见激活函数

### 1. ReLU

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

例如：

$$
x=
\begin{bmatrix}
2\\
-1\\
3\\
-4
\end{bmatrix}
$$

经过 ReLU：

$$
\mathrm{ReLU}(x)=
\begin{bmatrix}
2\\
0\\
3\\
0
\end{bmatrix}
$$

ReLU 的导数为：

$$
\mathrm{ReLU}'(x)=
\begin{cases}
1,&x>0\\
0,&x<0
\end{cases}
$$

反向传播时：

- 正数位置允许梯度通过；
- 负数位置的梯度变为 0。

记忆：

$$
\boxed{
\text{ReLU 前向保留正数，反向保留正数位置的梯度}
}
$$

### 2. Sigmoid

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

输出范围为：

$$
0<\sigma(x)<1
$$

导数为：

$$
\sigma'(x)=\sigma(x)(1-\sigma(x))
$$

如果前向传播已经得到：

$$
\sigma(x)=0.73
$$

那么：

$$
\sigma'(x)
=
0.73(1-0.73)
\approx0.20
$$

当输入的绝对值很大时，Sigmoid 的导数接近 0，容易出现梯度消失。

因此，现代神经网络的隐藏层中通常更常使用 ReLU。

<a id="section-04"></a>

## 四、全连接网络中的维度

课程示例：

```python
N, D_in, H, D_out = 64, 1000, 100, 10
```

各变量含义如下：

| 符号 | 含义 | 示例值 |
|---|---|---:|
| `N` | 一个批次中的样本数量 | 64 |
| D<sub>in</sub> | 每个样本的输入特征数 | 1000 |
| `H` | 隐藏层神经元数 | 100 |
| D<sub>out</sub> | 每个样本的输出维度 | 10 |

输入数据为：

$$
X\in\mathbb{R}^{64\times1000}
$$

表示：

- 共有 64 个样本；
- 每个样本包含 1000 个特征。

因此：

```python
D_in = 1000
```

不是说输入只有数字 1000，而是：

$$
\boxed{
\text{每个样本由1000个数表示}
}
$$

<a id="section-05"></a>

## 五、输入维度如何确定

输入维度由一个样本包含的特征数量决定。

例如，一张灰度图像大小为：

$$
28\times28
$$

展开成一维向量后：

$$
D_{\text{in}}=28\times28=784
$$

一张 CIFAR-10 彩色图像大小为：

$$
32\times32\times3
$$

展开后：

$$
D_{\text{in}}=32\times32\times3=3072
$$

因此：

$$
\boxed{
D_{\text{in}}
=
\text{一个样本展开后包含的数值总数}
}
$$

<a id="section-06"></a>

## 六、隐藏层神经元个数可以不同吗

可以。

例如：

$$
1000\rightarrow100\rightarrow10
$$

也可以设计成：

$$
1000\rightarrow200\rightarrow50\rightarrow10
$$

隐藏层神经元数量属于超参数，由设计者选择。

一般来说：

- 神经元太少：表达能力不足，容易欠拟合；
- 神经元太多：参数量增加，计算变慢，也可能过拟合。

唯一必须满足的是相邻矩阵的维度能够相乘。

例如：

$$
X\in\mathbb{R}^{64\times1000}
$$

第一隐藏层有 100 个神经元，则：

$$
W_1\in\mathbb{R}^{1000\times100}
$$

矩阵乘法为：

$$
XW_1:
(64\times1000)(1000\times100)
=
64\times100
$$

所以隐藏层输出为：

$$
H\in\mathbb{R}^{64\times100}
$$

<a id="section-07"></a>

## 七、内积与神经元计算

两个同维向量的内积为：

$$
a^Tb=\sum_{i=1}^{n}a_ib_i
$$

也就是：

$$
\boxed{
\text{对应元素相乘，再把结果全部相加}
}
$$

例如：

$$
a=
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix},
\qquad
b=
\begin{bmatrix}
4\\
5\\
6
\end{bmatrix}
$$

则：

$$
a^Tb
=
1\times4+2\times5+3\times6
=32
$$

一个神经元的基本计算为：

$$
z=w^Tx+b
$$

例如：

$$
w=
\begin{bmatrix}
0.5\\
-1\\
2
\end{bmatrix},
\qquad
x=
\begin{bmatrix}
2\\
3\\
4
\end{bmatrix},
\qquad
b=1
$$

先计算内积：

$$
w^Tx
=
0.5\times2+(-1)\times3+2\times4
=6
$$

再加上偏置：

$$
z=6+1=7
$$

因此一个神经元完成的过程是：

$$
\boxed{
\text{输入加权求和}
\rightarrow
\text{加偏置}
\rightarrow
\text{经过激活函数}
}
$$

<a id="section-08"></a>

## 八、两层神经网络的前向传播

设：

$$
X\in\mathbb{R}^{N\times D_{\text{in}}}
$$

$$
W_1\in\mathbb{R}^{D_{\text{in}}\times H}
$$

$$
W_2\in\mathbb{R}^{H\times D_{\text{out}}}
$$

前向传播为：

$$
Z_1=XW_1+b_1
$$

$$
H=f(Z_1)
$$

$$
Y_{\text{pred}}=HW_2+b_2
$$

对应代码：

```python
z1 = x.dot(w1) + b1
h = np.maximum(0, z1)
y_pred = h.dot(w2) + b2
```

维度变化为：

$$
N\times D_{\text{in}}
\rightarrow
N\times H
\rightarrow
N\times D_{\text{out}}
$$

例如：

$$
64\times1000
\rightarrow
64\times100
\rightarrow
64\times10
$$

<a id="section-09"></a>

## 九、什么是计算图

计算图是将复杂函数拆成多个简单运算。

例如：

$$
f(x,y,z)=(x+y)z
$$

可以拆成：

$$
q=x+y
$$

$$
f=qz
$$

前向传播从左向右计算数值：

$$
x,y,z
\rightarrow
q
\rightarrow
f
$$

反向传播从右向左计算梯度：

$$
f
\rightarrow
q,z
\rightarrow
x,y
$$

计算图的优点是：

> 不需要直接对整个复杂函数一次性求导，只需要计算每个简单节点的局部梯度。

<a id="section-10"></a>

## 十、反向传播的核心：链式法则

假设：

$$
x\rightarrow y\rightarrow L
$$

其中：

$$
y=f(x)
$$

根据链式法则：

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial x}
$$

其中：

- $\frac{\partial L}{\partial y}$：上游梯度；
- $\frac{\partial y}{\partial x}$：局部梯度；
- $\frac{\partial L}{\partial x}$：传给前一个节点的梯度。

因此可以记为：

$$
\boxed{
\text{传回梯度}
=
\text{上游梯度}
\times
\text{局部梯度}
}
$$

### 例子

设：

$$
y=x^2
$$

$$
L=3y
$$

当：

$$
x=2
$$

局部梯度为：

$$
\frac{\partial y}{\partial x}=2x=4
$$

上游梯度为：

$$
\frac{\partial L}{\partial y}=3
$$

因此：

$$
\frac{\partial L}{\partial x}
=
3\times4
=12
$$

<a id="section-11"></a>

## 十一、常见节点的反向传播规律

### 1. 加法节点：梯度原样分发

前向传播：

$$
z=x+y
$$

局部梯度：

$$
\frac{\partial z}{\partial x}=1
$$

$$
\frac{\partial z}{\partial y}=1
$$

如果上游梯度为 $g$，那么：

$$
\frac{\partial L}{\partial x}=g
$$

$$
\frac{\partial L}{\partial y}=g
$$

记忆：

$$
\boxed{
\text{加法节点：上游梯度原样传给每个输入}
}
$$

### 2. 乘法节点：乘上另一个输入

前向传播：

$$
z=xy
$$

局部梯度：

$$
\frac{\partial z}{\partial x}=y
$$

$$
\frac{\partial z}{\partial y}=x
$$

如果上游梯度为 $g$，则：

$$
\frac{\partial L}{\partial x}=gy
$$

$$
\frac{\partial L}{\partial y}=gx
$$

例如：

$$
x=2,\qquad y=3,\qquad g=5
$$

则：

$$
\frac{\partial L}{\partial x}
=
5\times3
=15
$$

$$
\frac{\partial L}{\partial y}
=
5\times2
=10
$$

记忆：

$$
\boxed{
\text{乘法节点：乘上另一个输入}
}
$$

### 3. 复制节点：梯度相加

如果同一个变量 $x$ 在前向传播中进入两条路径：

$$
x\rightarrow y_1
$$

$$
x\rightarrow y_2
$$

那么反向传播到 $x$ 时，需要把两条路径传回来的梯度相加：

$$
\frac{\partial L}{\partial x}
=
\left(
\frac{\partial L}{\partial x}
\right)_{\text{路径1}}
+
\left(
\frac{\partial L}{\partial x}
\right)_{\text{路径2}}
$$

例如两条路径分别传回：

$$
4,\qquad2
$$

则：

$$
\frac{\partial L}{\partial x}
=
4+2
=6
$$

记忆：

$$
\boxed{
\text{前向复制，反向相加}
}
$$

注意：

> 只有多个梯度都属于同一个变量时，才进行相加。

### 4. Max 节点：梯度传给最大值

前向传播：

$$
z=\max(x,y)
$$

假设：

$$
x=4,\qquad y=5
$$

由于最大值来自 $y$，所以：

$$
\frac{\partial z}{\partial x}=0
$$

$$
\frac{\partial z}{\partial y}=1
$$

如果上游梯度为 $g$，则：

$$
\frac{\partial L}{\partial x}=0
$$

$$
\frac{\partial L}{\partial y}=g
$$

记忆：

$$
\boxed{
\text{Max 节点：梯度传给前向传播中的获胜者}
}
$$

ReLU：

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

本质上就是一个 Max 节点。

### 5. 倒数节点

对于：

$$
f(x)=\frac{1}{x}
$$

导数为：

$$
\frac{df}{dx}
=
-\frac{1}{x^2}
$$

当：

$$
x=1.37
$$

局部梯度为：

$$
-\frac{1}{1.37^2}
\approx-0.53
$$

如果上游梯度为 1，则传回的梯度为：

$$
1\times(-0.53)
=
-0.53
$$

负号表示：

> 输入增大时，倒数函数的输出会减小。

<a id="section-12"></a>

## 十二、什么是雅可比矩阵

如果输入和输出都是向量：

$$
x\in\mathbb{R}^n
$$

$$
y=f(x)\in\mathbb{R}^m
$$

那么 $y$ 对 $x$ 的导数称为雅可比矩阵：

$$
J=\frac{\partial y}{\partial x}
$$

其中：

$$
J_{ij}
=
\frac{\partial y_i}{\partial x_j}
$$

展开为：

$$
J=
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1}
&
\cdots
&
\frac{\partial y_1}{\partial x_n}
\\
\vdots
&
\ddots
&
\vdots
\\
\frac{\partial y_m}{\partial x_1}
&
\cdots
&
\frac{\partial y_m}{\partial x_n}
\end{bmatrix}
$$

雅可比矩阵的维度为：

$$
\boxed{
m\times n
=
\text{输出维度}\times\text{输入维度}
}
$$

它记录：

> 每一个输出分别受到每一个输入多大的影响。

<a id="section-13"></a>

## 十三、雅可比矩阵例子

设：

$$
y_1=x_1^2+x_2
$$

$$
y_2=x_1x_2
$$

分别求偏导：

$$
\frac{\partial y_1}{\partial x_1}=2x_1
$$

$$
\frac{\partial y_1}{\partial x_2}=1
$$

$$
\frac{\partial y_2}{\partial x_1}=x_2
$$

$$
\frac{\partial y_2}{\partial x_2}=x_1
$$

所以：

$$
J=
\begin{bmatrix}
2x_1&1\\
x_2&x_1
\end{bmatrix}
$$

当输入发生小变化 $\Delta x$ 时：

$$
\Delta y\approx J\Delta x
$$

因此，雅可比矩阵可以理解为函数在当前点附近的局部线性变换。

<a id="section-14"></a>

## 十四、线性层的雅可比矩阵

对于：

$$
y=Wx+b
$$

展开第 $i$ 个输出：

$$
y_i=\sum_jW_{ij}x_j+b_i
$$

因此：

$$
\frac{\partial y_i}{\partial x_j}=W_{ij}
$$

所以：

$$
\boxed{
\frac{\partial y}{\partial x}=W
}
$$

线性层对输入的雅可比矩阵就是权重矩阵本身。

<a id="section-15"></a>

## 十五、ReLU 的雅可比矩阵

设：

$$
x=
\begin{bmatrix}
1\\
-2\\
3\\
-1
\end{bmatrix}
$$

经过 ReLU：

$$
z=
\begin{bmatrix}
1\\
0\\
3\\
0
\end{bmatrix}
$$

ReLU 的雅可比矩阵为：

$$
\frac{\partial z}{\partial x}
=
\begin{bmatrix}
1&0&0&0\\
0&0&0&0\\
0&0&1&0\\
0&0&0&0
\end{bmatrix}
$$

假设上游梯度为：

$$
\frac{\partial L}{\partial z}
=
\begin{bmatrix}
4\\
-1\\
5\\
9
\end{bmatrix}
$$

则：

$$
\frac{\partial L}{\partial x}
=
\begin{bmatrix}
4\\
0\\
5\\
0
\end{bmatrix}
$$

实际程序中不需要构造完整雅可比矩阵，可以直接写成：

```python
dx = dout * (x > 0)
```

其中 `(x > 0)` 会生成由 0 和 1 构成的掩码。

<a id="section-16"></a>

## 十六、矩阵乘法的反向传播

设：

$$
Y=XW
$$

其中：

$$
X\in\mathbb{R}^{N\times D}
$$

$$
W\in\mathbb{R}^{D\times M}
$$

$$
Y\in\mathbb{R}^{N\times M}
$$

假设上游梯度为：

$$
G=\frac{\partial L}{\partial Y}
$$

则：

$$
\boxed{
\frac{\partial L}{\partial X}
=
GW^T
}
$$

$$
\boxed{
\frac{\partial L}{\partial W}
=
X^TG
}
$$

### 输入梯度的维度检查

$$
GW^T:
(N\times M)(M\times D)
=
N\times D
$$

所以：

$$
\frac{\partial L}{\partial X}
$$

与 $X$ 的形状相同。

### 权重梯度的维度检查

$$
X^TG:
(D\times N)(N\times M)
=
D\times M
$$

所以：

$$
\frac{\partial L}{\partial W}
$$

与 $W$ 的形状相同。

重要规律：

$$
\boxed{
\text{一个变量的梯度必须与该变量的形状相同}
}
$$

<a id="section-17"></a>

## 十七、两层神经网络的反向传播

前向传播：

$$
Z_1=XW_1
$$

$$
H=\mathrm{ReLU}(Z_1)
$$

$$
Y_{\text{pred}}=HW_2
$$

假设使用平方损失：

$$
L=\sum(Y_{\text{pred}}-Y)^2
$$

反向传播从右向左进行。

### 第一步：损失对预测结果求导

$$
\frac{\partial L}{\partial Y_{\text{pred}}}
=
2(Y_{\text{pred}}-Y)
$$

代码：

```python
grad_y_pred = 2.0 * (y_pred - y)
```

### 第二步：计算第二层权重梯度

由于：

$$
Y_{\text{pred}}=HW_2
$$

所以：

$$
\frac{\partial L}{\partial W_2}
=
H^T
\frac{\partial L}{\partial Y_{\text{pred}}}
$$

代码：

```python
grad_w2 = h.T.dot(grad_y_pred)
```

### 第三步：把梯度传回隐藏层

$$
\frac{\partial L}{\partial H}
=
\frac{\partial L}{\partial Y_{\text{pred}}}W_2^T
$$

代码：

```python
grad_h = grad_y_pred.dot(w2.T)
```

### 第四步：经过 ReLU 节点

$$
H=\mathrm{ReLU}(Z_1)
$$

所以：

$$
\frac{\partial L}{\partial Z_1}
=
\frac{\partial L}{\partial H}
\odot
\mathbf{1}_{Z_1>0}
$$

其中 $\odot$ 表示逐元素相乘。

代码：

```python
grad_z1 = grad_h.copy()
grad_z1[z1 < 0] = 0
```

### 第五步：计算第一层权重梯度

由于：

$$
Z_1=XW_1
$$

所以：

$$
\frac{\partial L}{\partial W_1}
=
X^T
\frac{\partial L}{\partial Z_1}
$$

代码：

```python
grad_w1 = x.T.dot(grad_z1)
```

<a id="section-18"></a>

## 十八、完整训练代码

<details>
<summary><strong>展开查看 NumPy 完整实现</strong></summary>

```python
import numpy as np

# 数据维度
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机生成数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# 随机初始化权重
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6

for t in range(500):
    # 1. 前向传播
    z1 = x.dot(w1)
    h = np.maximum(z1, 0)
    y_pred = h.dot(w2)

    # 2. 计算平方损失
    loss = np.square(y_pred - y).sum()

    if t % 100 == 0:
        print(t, loss)

    # 3. 反向传播
    grad_y_pred = 2.0 * (y_pred - y)

    grad_w2 = h.T.dot(grad_y_pred)

    grad_h = grad_y_pred.dot(w2.T)

    grad_z1 = grad_h.copy()
    grad_z1[z1 < 0] = 0

    grad_w1 = x.T.dot(grad_z1)

    # 4. 更新参数
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```

训练过程可以理解为：

$$
\boxed{
\text{先预测}
\rightarrow
\text{发现误差}
\rightarrow
\text{计算每个参数的责任}
\rightarrow
\text{修改参数}
}
$$

</details>

<a id="section-19"></a>

## 十九、为什么不直接构造完整雅可比矩阵

大型神经网络可能包含数百万个输入和输出。

如果显式构造：

$$
\frac{\partial Y}{\partial X}
$$

雅可比矩阵会非常巨大，占用大量内存。

反向传播真正需要的通常不是完整雅可比矩阵，而是：

$$
J^Tg
$$

即雅可比矩阵与上游梯度的乘积。

因此，自动微分框架通常：

1. 前向传播时保存必要的中间变量；
2. 反向传播时直接计算梯度乘积；
3. 不显式构造完整雅可比矩阵。

<a id="section-20"></a>

## 二十、不同卷积滤波器之间有什么区别

对于输入图像：

$$
3\times32\times32
$$

假设使用 6 个滤波器，每个滤波器大小为：

$$
3\times5\times5
$$

这 6 个滤波器的尺寸相同，但内部的权重数值不同。

每个滤波器包含：

$$
3\times5\times5=75
$$

个权重。

不同滤波器可能分别学习：

- 水平边缘；
- 垂直边缘；
- 斜线；
- 颜色变化；
- 明暗变化；
- 局部纹理。

因此：

$$
\boxed{
\text{滤波器尺寸相同，但权重不同，所以提取的特征不同}
}
$$

<a id="section-21"></a>

## 二十一、一个滤波器如何计算

滤波器在图像上截取一个局部区域：

$$
X_{\text{local}}
\in
\mathbb{R}^{3\times5\times5}
$$

滤波器为：

$$
K\in\mathbb{R}^{3\times5\times5}
$$

在一个位置上的输出为：

$$
z=
\sum_{c=1}^{3}
\sum_{i=1}^{5}
\sum_{j=1}^{5}
K_{cij}X_{cij}+b
$$

也就是：

> 对应位置相乘，再将 75 个乘积全部相加。

一个滤波器在整张图像上滑动，会生成一张特征图。

因此：

$$
\boxed{
1\text{个滤波器}
\rightarrow
1\text{张特征图}
}
$$

6 个滤波器会生成 6 张特征图，所以输出通道数为 6：

$$
6\times28\times28
$$

因此：

$$
\boxed{
\text{滤波器数量}
=
\text{输出特征图数量}
=
\text{输出通道数}
}
$$

<a id="section-22"></a>

## 二十二、容易混淆的问题

### 1. 隐藏层的神经元数量可以不同吗

可以。

例如：

$$
1000\rightarrow200\rightarrow50\rightarrow10
$$

是完全合法的网络结构。

只需要保证相邻矩阵维度匹配。

### 2. `D_in = 1000` 表示输入只有一个数吗

不是。

它表示每个样本有 1000 个输入特征。

真正的输入数据可能是：

$$
X\in\mathbb{R}^{64\times1000}
$$

即 64 个样本，每个样本有 1000 个数。

### 3. 反向传播时什么时候把梯度相加

当同一个变量通过多条路径影响损失时，各路径传回的梯度需要相加：

$$
\boxed{
\text{同一变量，多条路径，梯度求和}
}
$$

### 4. 为什么矩阵反向传播中经常出现转置

因为梯度的形状必须与原变量一致。

例如：

$$
Y=XW
$$

传回 $X$ 的梯度为：

$$
\frac{\partial L}{\partial X}
=
\frac{\partial L}{\partial Y}W^T
$$

这里使用 $W^T$，可以保证结果的形状与 $X$ 相同。

### 5. 内积和逐元素相乘有什么区别

逐元素相乘：

$$
a\odot b
=
\begin{bmatrix}
a_1b_1\\
a_2b_2\\
\vdots
\end{bmatrix}
$$

结果仍然是向量。

内积：

$$
a^Tb
=
a_1b_1+a_2b_2+\cdots
$$

结果是一个标量。

因此：

$$
\boxed{
\text{内积}
=
\text{逐元素相乘后再求和}
}
$$

<a id="section-23"></a>

## 二十三、本讲核心总结

### 1. 神经网络的基本结构

$$
\boxed{
\text{线性变换}
+
\text{非线性激活}
}
$$

激活函数使多层网络不再等价于一个线性模型。

### 2. 一个神经元的计算

$$
z=w^Tx+b
$$

即：

$$
\boxed{
\text{权重与输入做内积，再加偏置}
}
$$

### 3. 前向传播

从输入开始，逐层计算：

$$
X
\rightarrow
Z_1
\rightarrow
H
\rightarrow
Y_{\text{pred}}
\rightarrow
L
$$

### 4. 反向传播

从损失开始，利用链式法则反向计算梯度：

$$
\boxed{
\text{传回梯度}
=
\text{上游梯度}
\times
\text{局部梯度}
}
$$

### 5. 常见节点规律

- 加法节点：梯度原样分发；
- 乘法节点：乘以另一个输入；
- 复制节点：多条路径的梯度相加；
- Max 节点：梯度传给最大值；
- ReLU 节点：正数位置传递梯度，负数位置截断梯度；
- 倒数节点：局部梯度为 $-1/x^2$。

### 6. 矩阵反向传播

对于：

$$
Y=XW
$$

有：

$$
\boxed{
\frac{\partial L}{\partial X}
=
\frac{\partial L}{\partial Y}W^T
}
$$

$$
\boxed{
\frac{\partial L}{\partial W}
=
X^T\frac{\partial L}{\partial Y}
}
$$

### 7. 梯度形状检查

$$
\frac{\partial L}{\partial X}
$$

必须与 $X$ 的形状相同。

$$
\frac{\partial L}{\partial W}
$$

必须与 $W$ 的形状相同。

### 8. 一句话理解反向传播

> 前向传播负责计算预测结果，反向传播负责判断每个参数对最终误差承担了多大责任。

## 仓库中的推荐文件结构

```text
CS231n-2025-Notes/
├── 03-regularization-optimization-gradient-descent.md
├── 04-neural-networks-and-backpropagation.md
└── README.md
```

> 本笔记不依赖外部图片，单独上传 Markdown 文件即可正常阅读。
