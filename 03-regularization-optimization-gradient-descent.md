# CS231n 2025 学习笔记（三）：正则化、优化与梯度下降

本节课主要解决两个问题：

1. 如何防止模型只记住训练数据，也就是防止过拟合？
2. 如何更新模型参数，使损失函数不断减小？

对应的两个核心主题是：

- **Regularization：正则化**
- **Optimization：优化**

## 一、从上一讲到这一讲

对于一个线性分类器，可以写成：

```math
\mathbf{s}
=
f(\mathbf{x};W,\mathbf{b})
=
W\mathbf{x}+\mathbf{b}
```

其中：

- $\mathbf{x}$：输入图像；
- $W$：权重矩阵；
- $\mathbf{b}$：偏置向量；
- $\mathbf{s}$：模型对不同类别给出的分数。

例如，模型对一张图片输出：

```math
\mathbf{s}
=
\begin{bmatrix}
2.1\\
5.3\\
-1.2
\end{bmatrix}
```

假设三个类别依次是猫、狗、汽车，那么“狗”的分数最高，因此模型会把这张图片预测为狗。

损失函数用于衡量模型当前的预测有多差：

```math
L(W,\mathbf{b})
```

训练模型的目标，是找到一组使损失函数尽可能小的参数：

```math
W^*,\mathbf{b}^*
=
\arg\min_{W,\mathbf{b}}
L(W,\mathbf{b})
```

但是，只追求训练损失变小会带来两个问题：

1. 模型可能过拟合训练集；
2. 模型参数数量很多，不能依靠随机尝试寻找最优参数。

因此，本节课分成两个主要部分：

```text
正则化：限制模型复杂度，减少过拟合

优化：更新模型参数，使损失函数下降
```

## 二、正则化 Regularization

### 2.1 什么是过拟合？

过拟合是指：

> 模型在训练数据上表现很好，但在没有见过的新数据上表现较差。

例如，训练集中的猫图片大多使用深色背景，模型可能错误地学到：

```text
深色背景 ⇒ 猫
```

它没有真正学习猫的耳朵、眼睛和轮廓等特征，而是记住了训练集中的偶然规律。

这种模型可能出现：

```text
训练集准确率很高

验证集或测试集准确率较低
```

说明模型的**泛化能力**较差。

泛化能力是指：

> 模型在没有参与训练的新数据上仍然保持良好性能的能力。

### 2.2 加入正则化后的损失函数

加入正则化后，总损失函数可以写成：

```math
L(W)
=
L_{\mathrm{data}}(W)
+
\lambda R(W)
```

其中：

- $L_{\mathrm{data}}(W)$：数据损失，要求模型正确拟合训练数据；
- $R(W)$：正则化项，用于限制模型复杂度；
- $\lambda$：正则化强度。

可以理解为模型同时完成两个任务：

```text
数据损失：要求模型预测准确

正则化项：要求模型不要过于复杂
```

因此，总目标是：

```text
既要拟合训练数据

又要避免参数过度复杂
```

### 2.3 正则化系数 λ

$\lambda$ 用于控制正则化项的重要程度。

当 $\lambda$ 很小时：

- 模型主要追求训练集上的预测准确率；
- 对参数复杂度限制较弱；
- 容易出现过拟合。

当 $\lambda$ 很大时：

- 模型会过度限制参数；
- 可能无法充分拟合训练数据；
- 容易出现欠拟合。

因此：

> $\lambda$ 是一个超参数，通常需要通过验证集选择。

## 三、L1、L2 与 Elastic Net 正则化

### 3.1 L2 正则化

L2 正则项通常写成：

```math
R_{L2}(W)
=
\sum_{i,j}W_{ij}^{2}
=
\lVert W\rVert_2^2
```

加入 L2 正则化后的总损失为：

```math
L(W)
=
L_{\mathrm{data}}(W)
+
\lambda\lVert W\rVert_2^2
```

L2 正则化会对绝对值较大的权重施加更强的惩罚。

例如，比较两组参数：

```math
\mathbf{w}_1=
\begin{bmatrix}
1\\
0\\
0\\
0
\end{bmatrix}
```

```math
\mathbf{w}_2=
\begin{bmatrix}
0.25\\
0.25\\
0.25\\
0.25
\end{bmatrix}
```

两组参数的元素之和都是 1，但是 L2 正则项不同。

对于第一组参数：

```math
\lVert\mathbf{w}_1\rVert_2^2
=
1^2+0^2+0^2+0^2
=
1
```

对于第二组参数：

```math
\lVert\mathbf{w}_2\rVert_2^2
=
4\times0.25^2
=
0.25
```

因此，L2 正则化更加倾向于选择第二组参数。

直观理解：

> L2 不希望模型只依赖某一个特别大的权重，而希望多个特征共同参与判断。

L2 正则化的特点：

- 抑制过大的权重；
- 使权重整体变小；
- 使不同特征共同参与预测；
- 通常不会使大量权重精确等于 0；
- 在神经网络中应用广泛。

### 3.2 L1 正则化

L1 正则项写成：

```math
R_{L1}(W)
=
\sum_{i,j}|W_{ij}|
=
\lVert W\rVert_1
```

加入 L1 正则化后的总损失为：

```math
L(W)
=
L_{\mathrm{data}}(W)
+
\lambda\lVert W\rVert_1
```

L1 正则化容易把一部分不重要的参数压缩为 0。

例如，原来的权重为：

```math
\mathbf{w}
=
\begin{bmatrix}
0.8\\
0.03\\
-0.01\\
0.6
\end{bmatrix}
```

经过 L1 正则化后，可能变成：

```math
\mathbf{w}
=
\begin{bmatrix}
0.7\\
0\\
0\\
0.5
\end{bmatrix}
```

模型只保留少量重要特征。

这种大量参数为 0 的特性称为：

> 稀疏性 Sparse

L1 正则化的特点：

- 使部分参数直接变为 0；
- 可以进行一定程度的特征选择；
- 产生稀疏模型；
- 便于模型压缩和解释。

### 3.3 为什么 L1 更容易产生 0？

对于单个权重 $w$，L1 正则项为：

```math
R_{L1}(w)=|w|
```

当 $w\neq0$ 时，它的导数为：

```math
\frac{\mathrm{d}|w|}{\mathrm{d}w}
=
\begin{cases}
1,&w>0\\
-1,&w<0
\end{cases}
```

无论权重本身有多小，L1 都会施加一个大小基本固定、方向指向 0 的作用。

因此，较小的权重容易被直接推到 0。

在 $w=0$ 处，绝对值函数不可导，实际优化中通常使用次梯度处理。

对于 L2 正则项：

```math
R_{L2}(w)=w^2
```

它的导数为：

```math
\frac{\mathrm{d}w^2}{\mathrm{d}w}
=
2w
```

当 $w$ 越接近 0 时，L2 的梯度也越小。

因此，L2 通常会使权重逐渐接近 0，但不容易让权重精确等于 0。

### 3.4 L1 与 L2 的区别

| 方法 | 正则项 | 对权重的主要影响 |
|---|---|---|
| L1 正则化 | 权重绝对值之和 | 使部分权重直接变为 0，产生稀疏模型 |
| L2 正则化 | 权重平方和 | 抑制较大的权重，使权重整体变小、更加分散 |
| Elastic Net | L1 与 L2 的组合 | 同时获得稀疏性和权重平滑效果 |

可以简单记为：

```text
L1：删除不重要的特征

L2：不让某个特征的权重过大
```

### 3.5 Elastic Net

Elastic Net 同时使用 L1 和 L2 正则化：

```math
R(W)
=
\lambda_1\lVert W\rVert_1
+
\lambda_2\lVert W\rVert_2^2
```

总损失为：

```math
L(W)
=
L_{\mathrm{data}}(W)
+
\lambda_1\lVert W\rVert_1
+
\lambda_2\lVert W\rVert_2^2
```

其中：

- $\lambda_1$ 控制 L1 正则化强度；
- $\lambda_2$ 控制 L2 正则化强度。

Elastic Net 同时具有：

- L1 的稀疏性；
- L2 的平滑性和稳定性。

## 四、优化 Optimization

优化的目标是寻找使损失函数最小的参数：

```math
W^*
=
\arg\min_W L(W)
```

最简单的方法是随机生成很多组 $W$，计算每一组参数对应的损失，再选择损失最小的一组。

但是，神经网络可能包含数百万、数十亿甚至更多参数，随机搜索几乎不可能找到理想结果。

因此，我们需要知道：

> 当前参数应该向哪个方向变化，才能使损失函数下降？

这个方向由**梯度**给出。

## 五、梯度是什么？

### 5.1 从一元函数的导数开始

假设：

```math
f(x)=x^2
```

它的导数为：

```math
f'(x)=2x
```

在 $x=3$ 处：

```math
f'(3)=6
```

它表示：

> 在 $x=3$ 附近，$x$ 增加一个很小的量时，函数值大约以 6 倍的速度增加。

导数主要包含两个信息：

1. 正负号表示函数增加还是减小；
2. 绝对值表示函数变化速度有多快。

### 5.2 多变量函数的梯度

如果函数有多个变量，例如：

```math
f(x,y)=x^2+3y^2
```

就分别对每个变量求偏导数：

```math
\frac{\partial f}{\partial x}=2x
```

```math
\frac{\partial f}{\partial y}=6y
```

把所有偏导数组成一个向量，就是梯度：

```math
\nabla f(x,y)
=
\begin{bmatrix}
2x\\
6y
\end{bmatrix}
```

在点 $(1,2)$ 处：

```math
\nabla f(1,2)
=
\begin{bmatrix}
2\\
12
\end{bmatrix}
```

这说明在当前位置：

- 沿 $x$ 正方向，函数变化率为 2；
- 沿 $y$ 正方向，函数变化率为 12；
- 函数对 $y$ 方向的变化更加敏感。

## 六、梯度的几何意义

### 6.1 梯度方向

梯度方向是函数值上升最快的方向：

```math
\nabla f(\mathbf{x})
```

负梯度方向是函数值下降最快的方向：

```math
-\nabla f(\mathbf{x})
```

因此，梯度下降使用：

```math
\mathbf{x}_{t+1}
=
\mathbf{x}_t
-
\eta\nabla f(\mathbf{x}_t)
```

而不是加上梯度。

### 6.2 梯度大小

梯度的模为：

```math
\lVert\nabla f\rVert_2
=
\sqrt{
\left(\frac{\partial f}{\partial x_1}\right)^2
+\cdots+
\left(\frac{\partial f}{\partial x_n}\right)^2
}
```

它表示函数在当前位置最陡方向上的变化速度。

例如：

```math
\nabla f=
\begin{bmatrix}
3\\
4
\end{bmatrix}
```

梯度大小为：

```math
\lVert\nabla f\rVert_2
=
\sqrt{3^2+4^2}
=
5
```

可以理解为当前位置最陡的坡度大小为 5。

### 6.3 为什么梯度垂直于等高线？

等高线上的所有点具有相同的函数值。

沿着等高线方向移动时：

```math
\mathrm{d}f=0
```

方向导数可以表示为：

```math
\mathrm{d}f
=
\nabla f^T\mathrm{d}\mathbf{x}
```

因此，在等高线方向上：

```math
\nabla f^T\mathrm{d}\mathbf{x}=0
```

两个向量的内积为 0，说明它们相互垂直。

所以：

> 梯度垂直于等高线，并指向函数值增长最快的方向。

## 七、梯度的物理意义

梯度通常表示：

> 某个标量物理量在空间中增长最快的方向，以及沿该方向的最大增长速度。

### 7.1 温度梯度

假设一个房间中的温度分布为：

```math
T(x,y,z)=2x+3z
```

温度梯度为：

```math
\nabla T
=
\begin{bmatrix}
\frac{\partial T}{\partial x}\\
\frac{\partial T}{\partial y}\\
\frac{\partial T}{\partial z}
\end{bmatrix}
=
\begin{bmatrix}
2\\
0\\
3
\end{bmatrix}
```

它的含义是：

- 沿 $x$ 正方向移动 1 个单位，温度约增加 2；
- 沿 $y$ 方向移动，温度不变；
- 沿 $z$ 正方向移动 1 个单位，温度约增加 3。

综合来看，温度升高最快的方向是：

```math
\begin{bmatrix}
2\\
0\\
3
\end{bmatrix}
```

### 7.2 为什么三个变量对应三个梯度分量？

如果函数为：

```math
f(x,y,z)
```

它有三个输入变量，因此梯度必须有三个分量：

```math
\nabla f
=
\begin{bmatrix}
\frac{\partial f}{\partial x}\\
\frac{\partial f}{\partial y}\\
\frac{\partial f}{\partial z}
\end{bmatrix}
```

每一个输入变量都对应一个偏导数。

即使函数不随某个变量变化，也要保留对应分量，并将其写成 0。

例如：

```math
f(x,y,z)=x^2
```

梯度为：

```math
\nabla f
=
\begin{bmatrix}
2x\\
0\\
0
\end{bmatrix}
```

不能因为函数只随 $x$ 变化，就把梯度写成一个数。

规律是：

```text
梯度的维度 = 函数输入变量的数量
```

### 7.3 势能与力

如果物体的势能为：

```math
U(\mathbf{x})
```

那么力通常为：

```math
\mathbf{F}
=
-\nabla U(\mathbf{x})
```

例如，弹簧势能为：

```math
U(x)=\frac{1}{2}kx^2
```

对 $x$ 求导：

```math
\frac{\mathrm{d}U}{\mathrm{d}x}
=
kx
```

因此弹簧力为：

```math
F=-kx
```

这就是胡克定律。

可以发现：

- 梯度指向势能增加最快的方向；
- 负梯度指向势能下降最快的方向；
- 力使物体向低势能方向运动。

这与神经网络优化非常相似：

```text
物体：沿负势能梯度运动

参数：沿负损失梯度更新
```

## 八、梯度怎么求？

求梯度的基本步骤是：

1. 确定函数包含哪些变量；
2. 分别对每个变量求偏导；
3. 将所有偏导数组成一个向量。

例如：

```math
f(x,y)=x^2+xy+y^2
```

对 $x$ 求偏导时，将 $y$ 当成常数：

```math
\frac{\partial f}{\partial x}
=
2x+y
```

对 $y$ 求偏导时，将 $x$ 当成常数：

```math
\frac{\partial f}{\partial y}
=
x+2y
```

因此：

```math
\nabla f(x,y)
=
\begin{bmatrix}
2x+y\\
x+2y
\end{bmatrix}
```

在点 $(1,2)$ 处：

```math
\nabla f(1,2)
=
\begin{bmatrix}
4\\
5
\end{bmatrix}
```

函数上升最快的方向是：

```math
\begin{bmatrix}
4\\
5
\end{bmatrix}
```

函数下降最快的方向是：

```math
\begin{bmatrix}
-4\\
-5
\end{bmatrix}
```

## 九、数值梯度与解析梯度

### 9.1 数值梯度

数值梯度通过给参数一个很小的变化，观察函数值的变化。

中心差分公式为：

```math
\frac{\partial f}{\partial x}
\approx
\frac{f(x+h)-f(x-h)}{2h}
```

其中 $h$ 很小，例如：

```math
h=10^{-5}
```

例子：

```math
f(x)=x^2
```

在 $x=3$ 处，取：

```math
h=10^{-5}
```

则：

```math
f'(3)
\approx
\frac{(3+h)^2-(3-h)^2}{2h}
```

结果约为：

```math
6
```

与解析导数：

```math
f'(3)=2\times3=6
```

基本一致。

数值梯度的特点：

- 容易实现；
- 只需要计算函数值；
- 得到的是近似值；
- 计算速度很慢；
- 主要用于检查梯度是否正确。

### 9.2 解析梯度

解析梯度通过微积分、链式法则和反向传播计算。

神经网络训练时主要使用解析梯度，因为它的计算速度更快。

实际中通常采用：

```text
解析梯度：用于真正训练模型

数值梯度：用于验证解析梯度是否正确
```

这种检查称为：

> Gradient Check，梯度检查

常见的相对误差为：

```math
\mathrm{relative\ error}
=
\frac{
|g_{\mathrm{analytic}}-g_{\mathrm{numeric}}|
}{
\max
\left(
1,
|g_{\mathrm{analytic}}|,
|g_{\mathrm{numeric}}|
\right)
}
```

相对误差越小，说明解析梯度与数值梯度越接近。

## 十、梯度下降 Gradient Descent

梯度下降的参数更新公式为：

```math
W_{t+1}
=
W_t
-
\eta\nabla_W L(W_t)
```

其中：

- $W_t$：当前参数；
- $\nabla_W L(W_t)$：当前梯度；
- $\eta$：学习率；
- $W_{t+1}$：更新后的参数。

最核心的理解是：

```text
梯度决定往哪个方向走

学习率决定一步走多远
```

梯度指向损失增加最快的方向，因此要沿负梯度方向更新。

## 十一、SGD：随机梯度下降

SGD 全称为：

```text
Stochastic Gradient Descent
```

中文是：

```text
随机梯度下降
```

### 11.1 为什么需要 SGD？

假设训练集有 $N$ 个样本，总损失为：

```math
L(W)
=
\frac{1}{N}
\sum_{i=1}^{N}L_i(W)
```

完整梯度为：

```math
\nabla_W L(W)
=
\frac{1}{N}
\sum_{i=1}^{N}
\nabla_W L_i(W)
```

如果训练集有 100 万张图片，每更新一次参数都计算全部图片，计算速度会非常慢。

因此，可以随机选取一小批样本：

```math
\mathcal{B}
=
\{i_1,i_2,\ldots,i_B\}
```

用这一批样本的平均梯度近似完整梯度：

```math
g_t
=
\frac{1}{B}
\sum_{i\in\mathcal{B}}
\nabla_W L_i(W_t)
```

然后更新参数：

```math
W_{t+1}
=
W_t-\eta g_t
```

### 11.2 三种梯度下降方式

#### Batch Gradient Descent

每次使用全部训练数据计算梯度。

优点：

- 梯度较准确；
- 更新方向稳定。

缺点：

- 计算量大；
- 更新速度慢；
- 大数据集可能无法一次加载到内存。

#### 纯 SGD

每次只使用一个样本计算梯度。

优点：

- 每次更新速度快；
- 内存占用小。

缺点：

- 梯度噪声大；
- 更新路径抖动明显；
- 训练过程不够稳定。

#### Mini-batch SGD

每次使用一小批数据，例如：

```text
32、64、128 或 256 个样本
```

Mini-batch SGD 兼顾了计算效率和梯度稳定性。

深度学习中所说的 SGD，通常实际指：

> Mini-batch SGD

## 十二、学习率怎么理解？

学习率通常记为：

```math
\eta
```

参数的实际更新量为：

```math
\Delta W
=
-\eta\nabla_W L
```

因此，学习率控制当前梯度对参数的影响程度。

### 12.1 数值例子

假设当前参数为：

```math
w=5
```

当前梯度为：

```math
\frac{\partial L}{\partial w}=2
```

当学习率为：

```math
\eta=0.1
```

参数更新为：

```math
w_{\mathrm{new}}
=
5-0.1\times2
=
4.8
```

参数移动了：

```math
0.2
```

当学习率为：

```math
\eta=0.01
```

参数更新为：

```math
w_{\mathrm{new}}
=
5-0.01\times2
=
4.98
```

参数只移动了：

```math
0.02
```

因此，在梯度相同的情况下：

```text
学习率越大 → 参数更新幅度越大

学习率越小 → 参数更新幅度越小
```

### 12.2 学习率太小

学习率太小时：

- 参数每次变化很小；
- 损失下降速度慢；
- 训练时间长；
- 可能长时间停留在平坦区域。

### 12.3 学习率太大

学习率太大时：

- 参数可能一步跨过最低点；
- 参数在最低点两侧来回震荡；
- 损失函数可能忽高忽低；
- 严重时会发生发散；
- 可能出现 `NaN`。

### 12.4 一个完整例子

假设损失函数为：

```math
L(w)=w^2
```

梯度为：

```math
\frac{\mathrm{d}L}{\mathrm{d}w}=2w
```

更新公式为：

```math
w_{t+1}
=
w_t-2\eta w_t
```

整理得到：

```math
w_{t+1}
=
(1-2\eta)w_t
```

#### 当 η = 0.1

```math
w_{t+1}=0.8w_t
```

参数稳定地向 0 靠近。

#### 当 η = 0.75

```math
w_{t+1}=-0.5w_t
```

参数会在 0 两侧震荡，但幅值逐渐减小，最终仍然收敛。

#### 当 η = 1.1

```math
w_{t+1}=-1.2w_t
```

参数每次更新都会改变正负，而且绝对值越来越大，因此发生发散。

## 十三、学习率是固定值还是变化值？

两种情况都可以。

### 13.1 固定学习率

整个训练过程都使用相同的学习率：

```math
\eta_t=\eta
```

这种方法简单，但固定学习率通常很难同时兼顾训练前期和训练后期。

### 13.2 变化学习率

训练前期使用较大的学习率，训练后期逐渐减小：

```math
\eta_1>\eta_2>\cdots>\eta_T
```

原因是：

- 训练前期距离最优点较远，需要快速移动；
- 训练后期接近最优点，需要小步调整；
- 后期学习率过大容易在最低点附近震荡。

可以类比停车：

```text
距离车位较远时：快速前进

接近车位时：逐渐减速

进入车位后：小幅调整
```

因此，实际训练通常采用：

```text
较大的初始学习率 + 逐渐衰减的学习率
```

## 十四、SGD 为什么会震荡？

考虑损失函数：

```math
L(x,y)=x^2+100y^2
```

其中：

- $x$ 方向比较平缓；
- $y$ 方向非常陡峭。

梯度为：

```math
\nabla L(x,y)
=
\begin{bmatrix}
2x\\
200y
\end{bmatrix}
```

梯度下降更新为：

```math
x_{t+1}
=
x_t-2\eta x_t
```

```math
y_{t+1}
=
y_t-200\eta y_t
```

由于 $y$ 方向的梯度很大，参数很容易一次跨过谷底。

如果：

```math
1-200\eta<0
```

那么 $y$ 每次更新后都会改变正负：

```text
正 → 负 → 正 → 负
```

因此，参数会不断穿过山谷底部，产生“之”字形震荡。

核心原因是：

> 不同方向的曲率差异很大，同一个学习率无法同时适合所有方向。

如果直接减小学习率：

- 陡峭方向的震荡会减弱；
- 但是平缓方向原本梯度就很小，移动会变得更慢。

这就是普通 SGD 面临的典型问题：

```text
陡峭方向：反复震荡

平缓方向：前进缓慢
```

## 十五、Hessian 矩阵

### 15.1 Hessian 是什么？

梯度由一阶偏导数组成，Hessian 由二阶偏导数组成。

对于函数：

```math
f(x_1,x_2,\ldots,x_n)
```

Hessian 为：

```math
H
=
\nabla^2 f
=
\begin{bmatrix}
\frac{\partial^2f}{\partial x_1^2}
&
\frac{\partial^2f}{\partial x_1\partial x_2}
&
\cdots
&
\frac{\partial^2f}{\partial x_1\partial x_n}
\\
\frac{\partial^2f}{\partial x_2\partial x_1}
&
\frac{\partial^2f}{\partial x_2^2}
&
\cdots
&
\frac{\partial^2f}{\partial x_2\partial x_n}
\\
\vdots
&
\vdots
&
\ddots
&
\vdots
\\
\frac{\partial^2f}{\partial x_n\partial x_1}
&
\frac{\partial^2f}{\partial x_n\partial x_2}
&
\cdots
&
\frac{\partial^2f}{\partial x_n^2}
\end{bmatrix}
```

如果函数有 $n$ 个变量，Hessian 就是一个：

```math
n\times n
```

矩阵。

### 15.2 Hessian 怎么求？

例如：

```math
f(x,y)=x^2+3xy+2y^2
```

先求梯度：

```math
\nabla f
=
\begin{bmatrix}
2x+3y\\
3x+4y
\end{bmatrix}
```

再对梯度中的每个分量继续求偏导。

第一行：

```math
\frac{\partial^2f}{\partial x^2}=2
```

```math
\frac{\partial^2f}{\partial x\partial y}=3
```

第二行：

```math
\frac{\partial^2f}{\partial y\partial x}=3
```

```math
\frac{\partial^2f}{\partial y^2}=4
```

所以：

```math
H
=
\begin{bmatrix}
2&3\\
3&4
\end{bmatrix}
```

### 15.3 Hessian 的含义

梯度告诉我们：

> 函数值往哪个方向变化最快。

Hessian 告诉我们：

> 梯度如何变化，以及不同方向上的曲率大小。

简单理解：

```text
梯度：告诉我们往哪里走

Hessian：告诉我们不同方向有多陡、多弯
```

Hessian 的特征值可以描述不同方向上的曲率：

- 特征值大：该方向曲率大，比较陡；
- 特征值小：该方向曲率小，比较平；
- 特征值都为正：附近可能是局部最小值；
- 特征值都为负：附近可能是局部最大值；
- 特征值有正有负：附近可能是鞍点。

### 15.4 Hessian 条件数

对于正定 Hessian，条件数可以表示为：

```math
\kappa(H)
=
\frac{\lambda_{\max}}{\lambda_{\min}}
```

其中：

- $\lambda_{\max}$：最大特征值；
- $\lambda_{\min}$：最小正特征值。

当：

```math
\kappa(H)\gg1
```

说明不同方向的曲率差异很大。

此时损失函数的等高线通常非常狭长，SGD 容易出现：

- 在陡峭方向来回震荡；
- 在平缓方向移动缓慢。

## 十六、普通 SGD 的三个主要问题

### 16.1 不同方向的曲率差异大

结果是：

```text
陡峭方向震荡

平缓方向前进缓慢
```

### 16.2 鞍点和平坦区域

当：

```math
\nabla L(W)=0
```

参数更新量也为 0。

但是梯度为 0 的点不一定是最小值，也可能是：

- 局部最大值；
- 鞍点；
- 平坦区域。

例如：

```math
f(x,y)=x^2-y^2
```

梯度为：

```math
\nabla f
=
\begin{bmatrix}
2x\\
-2y
\end{bmatrix}
```

在原点：

```math
\nabla f(0,0)
=
\begin{bmatrix}
0\\
0
\end{bmatrix}
```

但是原点不是最小值，而是鞍点：

- 沿 $x$ 方向，函数向上弯；
- 沿 $y$ 方向，函数向下弯。

### 16.3 Mini-batch 梯度存在噪声

不同 mini-batch 中包含的数据不同，因此计算出的梯度也不同：

```math
g_t^{(1)}
\neq
g_t^{(2)}
\neq
g_t^{(3)}
```

所以参数更新方向会发生抖动。

这种噪声有时可以帮助模型离开鞍点或较差区域，但也会使训练过程不够稳定。

## 十七、Momentum

Momentum 中文叫作**动量法**。

普通 SGD 只使用当前梯度：

```math
W_{t+1}
=
W_t-\eta g_t
```

Momentum 会保留过去的运动趋势：

```math
v_t
=
\beta v_{t-1}
+
g_t
```

```math
W_{t+1}
=
W_t-\eta v_t
```

其中：

- $v_t$：当前速度或历史梯度的累积；
- $\beta$：动量系数；
- $\eta$：学习率；
- $g_t$：当前梯度。

常用的动量系数为：

```math
\beta=0.9
```

### 17.1 Momentum 怎么理解？

普通 SGD 相当于：

> 每走一步，都忘记之前向哪个方向走。

Momentum 相当于：

> 一个小球沿山坡向下滚动，会保留之前积累的速度。

如果连续多次沿同一个方向更新：

- 历史方向会不断累积；
- 参数沿稳定方向加速。

如果某个方向反复改变：

- 正负方向会相互抵消；
- 该方向的震荡会减弱。

### 17.2 Momentum 为什么能减少震荡？

假设连续两次梯度为：

```math
g_1=
\begin{bmatrix}
1\\
10
\end{bmatrix}
```

```math
g_2=
\begin{bmatrix}
1\\
-10
\end{bmatrix}
```

水平方向：

```text
1、1
```

方向始终一致，因此会不断累积，使水平方向前进加快。

竖直方向：

```text
10、-10
```

方向反复变化，因此在历史速度中相互抵消，震荡减小。

所以：

```text
稳定方向：动量不断累积，加速前进

震荡方向：正负更新相互抵消，减小震荡
```

### 17.3 数值例子

设：

```math
\beta=0.9
```

```math
\eta=0.1
```

初始速度：

```math
v_0=0
```

第一次梯度：

```math
g_1=2
```

则：

```math
v_1
=
0.9\times0+2
=
2
```

参数更新量为：

```math
-\eta v_1=-0.2
```

第二次梯度仍然为：

```math
g_2=2
```

则：

```math
v_2
=
0.9\times2+2
=
3.8
```

参数更新量为：

```math
-\eta v_2=-0.38
```

虽然两次梯度相同，但第二次更新幅度更大，说明 Momentum 会沿稳定方向加速。

如果第二次梯度反向：

```math
g_2=-2
```

则：

```math
v_2
=
0.9\times2-2
=
-0.2
```

历史速度与当前反向梯度接近抵消，因此参数不会立刻大幅反向运动。

### 17.4 学习率与 Momentum 的区别

学习率控制：

```text
当前更新的总体步长
```

动量系数控制：

```text
过去的更新方向保留多少
```

可以类比为：

```text
学习率：当前踩油门的力度

Momentum：汽车已经积累的速度
```

## 十八、RMSProp

RMSProp 的核心思想是：

> 根据每个参数近期梯度的大小，自动调整该参数的实际更新步长。

梯度平方的指数移动平均为：

```math
s_t
=
\rho s_{t-1}
+
(1-\rho)g_t^2
```

参数更新为：

```math
W_{t+1}
=
W_t
-
\eta
\frac{g_t}
{\sqrt{s_t}+\varepsilon}
```

其中：

- $s_t$：梯度平方的指数移动平均；
- $\rho$：衰减系数，常用 0.9；
- $\varepsilon$：防止分母为 0；
- 平方、开方和除法都是逐元素操作。

### 18.1 为什么要计算梯度平方？

假设梯度不断变化：

```text
10、-10、10、-10
```

直接求平均：

```math
\frac{10-10+10-10}{4}=0
```

会错误地认为梯度很小。

平方后：

```text
100、100、100、100
```

就能正确表示该方向上的梯度幅值很大。

所以 RMSProp 关注的是：

> 某个方向上近期梯度的幅值有多大，而不是梯度具体为正还是负。

### 18.2 RMSProp 如何调整步长？

RMSProp 的有效学习率可以近似理解为：

```math
\eta_{\mathrm{effective}}
=
\frac{\eta}{\sqrt{s_t}+\varepsilon}
```

在陡峭方向：

- 梯度大；
- $s_t$ 大；
- 分母大；
- 实际更新步长变小。

在平缓方向：

- 梯度小；
- $s_t$ 小；
- 分母较小；
- 保留相对较大的更新步长。

因此：

```text
陡峭方向：自动减小步长

平缓方向：保留较大步长
```

### 18.3 数值例子

假设两个方向的梯度为：

```math
g=
\begin{bmatrix}
1\\
100
\end{bmatrix}
```

普通 SGD 的学习率为：

```math
\eta=0.01
```

更新量为：

```math
\Delta W
=
-\eta g
=
\begin{bmatrix}
-0.01\\
-1
\end{bmatrix}
```

第二个方向更新过大，容易震荡。

假设 RMSProp 统计得到：

```math
s=
\begin{bmatrix}
1\\
10000
\end{bmatrix}
```

则：

```math
\sqrt{s}
=
\begin{bmatrix}
1\\
100
\end{bmatrix}
```

更新量约为：

```math
\Delta W
=
-0.01
\begin{bmatrix}
1/1\\
100/100
\end{bmatrix}
=
\begin{bmatrix}
-0.01\\
-0.01
\end{bmatrix}
```

原本相差 100 倍的更新量被缩放到相近范围。

## 十九、Adam

Adam 全称为：

```text
Adaptive Moment Estimation
```

可以理解为：

```text
Adam = Momentum + RMSProp + 偏差修正
```

Adam 同时记录：

1. 梯度的历史平均值；
2. 梯度平方的历史平均值。

### 19.1 一阶矩：记录方向

```math
m_t
=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
```

其中：

```math
\beta_1=0.9
```

较常见。

$m_t$ 类似 Momentum，用于平滑梯度方向。

### 19.2 二阶矩：记录梯度大小

```math
v_t
=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
```

其中：

```math
\beta_2=0.999
```

较常见。

$v_t$ 类似 RMSProp，用于调整每个参数的实际更新步长。

### 19.3 为什么需要偏差修正？

Adam 通常初始化：

```math
m_0=0,\qquad v_0=0
```

第一次计算：

```math
m_1
=
(1-\beta_1)g_1
```

当：

```math
\beta_1=0.9
```

有：

```math
m_1=0.1g_1
```

这比真实梯度小很多，因为初始值 0 把平均值拉低了。

因此需要偏差修正：

```math
\hat{m}_t
=
\frac{m_t}{1-\beta_1^t}
```

```math
\hat{v}_t
=
\frac{v_t}{1-\beta_2^t}
```

第一次迭代时：

```math
\hat{m}_1
=
\frac{0.1g_1}{1-0.9}
=
g_1
```

这样就修正了初始估计偏小的问题。

### 19.4 Adam 参数更新

Adam 的最终更新公式为：

```math
W_{t+1}
=
W_t
-
\eta
\frac{\hat{m}_t}
{\sqrt{\hat{v}_t}+\varepsilon}
```

其中：

- $\hat{m}_t$：决定总体更新方向；
- $\hat{v}_t$：调整每个参数的实际步长；
- $\eta$：基础学习率；
- $\varepsilon$：防止分母为 0。

可以简单理解为：

```text
Momentum：告诉 Adam 往哪个方向走

RMSProp：告诉 Adam 每个方向走多大一步

偏差修正：修正训练初期统计量偏小的问题
```

## 二十、SGD、Momentum、RMSProp 与 Adam 对比

| 优化器 | 使用的信息 | 核心作用 |
|---|---|---|
| SGD | 当前梯度 | 沿负梯度方向更新 |
| Momentum | 当前梯度和历史方向 | 加速稳定方向，减小震荡 |
| RMSProp | 当前梯度和历史梯度平方 | 自适应调整每个参数的步长 |
| Adam | 历史梯度和历史梯度平方 | 同时平滑方向并调整步长 |

最容易记忆的方法：

```text
SGD：只看现在

Momentum：记住过去的方向

RMSProp：观察过去的梯度大小

Adam：既记方向，又记大小
```

## 二十一、AdamW

AdamW 是 Adam 的常用改进版本。

普通 Adam 中，如果直接把 L2 正则化梯度加入总梯度，正则化产生的梯度也会进入 Adam 的一阶矩和二阶矩统计。

这会导致：

```text
权重衰减和自适应梯度更新发生耦合
```

AdamW 将权重衰减单独处理：

```math
W_{t+1}
=
W_t
-
\eta
\frac{\hat{m}_t}
{\sqrt{\hat{v}_t}+\varepsilon}
-
\eta\lambda W_t
```

也可以写成：

```math
W_{t+1}
=
(1-\eta\lambda)W_t
-
\eta
\frac{\hat{m}_t}
{\sqrt{\hat{v}_t}+\varepsilon}
```

可以理解为：

1. 先执行 Adam 的自适应更新；
2. 再单独让权重缩小一点。

这种方法称为：

> Decoupled Weight Decay，解耦权重衰减

Transformer 和大语言模型中通常使用 AdamW。

## 二十二、学习率调度

学习率随着训练过程变化的方法称为：

> Learning Rate Schedule，学习率调度

### 22.1 Step Decay

每隔一段时间减小学习率：

```math
\eta_t
=
\eta_0\gamma^{\lfloor t/T\rfloor}
```

例如：

```text
第 1～30 个 epoch：0.1

第 31～60 个 epoch：0.01

第 61 个 epoch 以后：0.001
```

### 22.2 Exponential Decay

指数衰减可以写成：

```math
\eta_t
=
\eta_0\gamma^t
```

其中：

- $\eta_0$：初始学习率；
- $\gamma$：衰减系数；
- $t$：当前训练步数。

### 22.3 Cosine Decay

余弦衰减可以写成：

```math
\eta_t
=
\eta_{\min}
+
\frac{1}{2}
(\eta_{\max}-\eta_{\min})
\left(
1+\cos\frac{\pi t}{T}
\right)
```

学习率按照余弦曲线平滑下降。

### 22.4 Warmup

训练初期不直接使用较大的学习率，而是让学习率逐渐增大：

```math
\eta_t
=
\eta_{\max}\frac{t}{T_{\mathrm{warmup}}}
```

然后再逐渐衰减。

完整过程通常是：

```text
较小学习率
    ↓
Warmup 逐渐增大
    ↓
达到设定的最大学习率
    ↓
逐渐衰减
```

Warmup 可以降低训练初期梯度不稳定导致损失爆炸的风险。

## 二十三、一阶优化与二阶优化

### 23.1 一阶优化

SGD、Momentum、RMSProp 和 Adam 主要使用一阶梯度：

```math
\nabla L(W)
```

因此，它们属于一阶优化方法。

### 23.2 二阶优化

二阶优化不仅使用梯度，还使用 Hessian。

损失函数在当前位置附近可以近似为：

```math
L(W+\Delta W)
\approx
L(W)
+
\nabla L(W)^T\Delta W
+
\frac{1}{2}
\Delta W^T
H(W)
\Delta W
```

牛顿法更新为：

```math
W_{t+1}
=
W_t
-
H(W_t)^{-1}
\nabla L(W_t)
```

Hessian 能够根据不同方向上的曲率自动调整步长。

### 23.3 为什么神经网络很少直接使用完整 Hessian？

如果模型有 $n$ 个参数，Hessian 的大小为：

```math
n\times n
```

总元素数量为：

```math
n^2
```

假设：

```math
n=10^8
```

那么 Hessian 有：

```math
10^{16}
```

个元素。

存储、计算和求逆的代价都非常高。

因此，深度学习通常使用：

- SGD；
- Momentum；
- RMSProp；
- Adam；
- AdamW；

而不是直接计算完整 Hessian。

## 二十四、PyTorch 中的优化器

### 24.1 SGD

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

### 24.2 SGD with Momentum

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

### 24.3 RMSProp

```python
optimizer = torch.optim.RMSprop(
    model.parameters(),
    lr=1e-3,
    alpha=0.9,
    eps=1e-8
)
```

### 24.4 Adam

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

### 24.5 AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2
)
```

### 24.6 一个简单的训练循环

```python
for images, labels in train_loader:
    optimizer.zero_grad()

    scores = model(images)
    loss = criterion(scores, labels)

    loss.backward()
    optimizer.step()
```

各步骤含义：

```text
optimizer.zero_grad()：清除上一轮梯度

model(images)：执行前向传播

criterion(...)：计算损失

loss.backward()：执行反向传播，计算梯度

optimizer.step()：根据梯度更新参数
```

## 二十五、本节课的知识关系

```text
训练神经网络
│
├── 正则化：防止过拟合
│   ├── L1：产生稀疏权重
│   ├── L2：让权重整体变小
│   ├── Elastic Net：L1 与 L2 的组合
│   └── 权重衰减：限制模型复杂度
│
└── 优化：让损失函数下降
    ├── 梯度：决定更新方向
    ├── 学习率：决定更新步长
    ├── SGD：使用当前梯度
    ├── Momentum：累积历史方向
    ├── RMSProp：自适应调整步长
    ├── Adam：方向平滑 + 自适应步长
    └── AdamW：Adam + 独立权重衰减
```

## 二十六、核心问题快速复习

### 1. 梯度是什么？

梯度是函数对所有输入变量的偏导数组成的向量：

```math
\nabla f
=
\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\frac{\partial f}{\partial x_2}\\
\vdots\\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
```

### 2. 梯度有什么意义？

```text
梯度方向：函数值上升最快的方向

负梯度方向：函数值下降最快的方向

梯度大小：最陡方向上的变化速度
```

### 3. 函数有三个变量，梯度为什么也应该有三个分量？

因为每个输入变量都对应一个偏导数：

```math
\nabla f(x,y,z)
=
\begin{bmatrix}
\frac{\partial f}{\partial x}\\
\frac{\partial f}{\partial y}\\
\frac{\partial f}{\partial z}
\end{bmatrix}
```

### 4. 学习率怎么理解？

学习率控制参数每次更新的步长：

```math
\Delta W=-\eta\nabla_W L
```

### 5. 学习率是固定值还是变化值？

两种都可以，但实际训练中经常使用：

```text
训练前期：较大学习率

训练后期：逐渐减小学习率
```

### 6. SGD 为什么会震荡？

因为陡峭方向上的梯度很大，参数一次更新容易跨过谷底，导致梯度符号反复改变。

### 7. Hessian 是什么？

Hessian 是由二阶偏导数组成的矩阵，用于描述函数在不同方向上的曲率。

### 8. Momentum 的作用是什么？

```text
累积稳定方向上的历史更新

抵消反复变化方向上的震荡
```

### 9. RMSProp 的作用是什么？

```text
根据每个参数近期梯度的大小

自适应调整该参数的实际更新步长
```

### 10. Adam 的作用是什么？

Adam 结合了：

```text
Momentum：平滑更新方向

RMSProp：自适应调整步长

偏差修正：改善训练初期的统计估计
```

可以概括为：

```text
Adam = 方向平滑 + 自适应步长 + 偏差修正
```

## 二十七、本节课最重要的十句话

1. 正则化用于限制模型复杂度，减少过拟合。
2. L1 正则化容易产生稀疏权重。
3. L2 正则化会使权重整体变小、更加分散。
4. 梯度由函数对各个输入变量的偏导数组成。
5. 梯度指向函数值上升最快的方向，负梯度指向下降最快的方向。
6. 梯度决定参数更新方向，学习率决定参数更新步长。
7. SGD 在陡峭方向容易震荡，在平缓方向前进缓慢。
8. Hessian 描述损失函数在不同方向上的曲率。
9. Momentum 记录历史方向，RMSProp 调整每个参数的实际步长。
10. Adam 同时结合方向平滑、自适应步长和偏差修正。

## 参考资料

1. Stanford CS231n Spring 2025 Lecture 3.
2. Stanford CS231n 2025 官方课程讲义。
3. CS231n Course Notes: Optimization.
4. PyTorch Optimizer Documentation.