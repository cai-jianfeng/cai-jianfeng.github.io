---
title: 'The Basic Knowledge of PyTorch Autograd'
date: 23-12-15
update: 24-05-04
permalink: /posts/2023/12/blog-code-pytorch-autograd/
star: superior
tags:
  - 深度学习基本知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要介绍了 PyTorch 的 autograd 机制及其具体实现方式。</p>

Torch Autograd
===

<p style="text-align:justify; text-justify:inter-ideograph;">训练一个模型的范式是构造模型 $\mathcal{M}$ 和数据 $\mathcal{D} = \{x_i,y_i\}_{1:N}$，
使用模型的前向过程计算 loss：<code style="color: #B58900">l = M.forward(xi,yi)</code>，然后使用<code style="color: #B58900">l.backward()</code>计算 gradient，
最后使用<code style="color: #B58900">optim.step()</code>更新模型参数。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">那么 PyTorch 是如何计算每个参数的梯度的，即 PyTorch 的自动求导机制(<code style="color: #B58900">torch.autograd</code>)？
通俗而言，<code style="color: #B58900">torch.autograd</code>在模型 forward 的同时构造了一个 computational graph: DAG (由 Function 组成)；
其中的叶子节点表示需要计算梯度的输入数据和模型参数 (即<code style="color: #B58900">.required_grad=Ture</code>)，而非叶子节点表示模型对这些输入数据和参数进行的初等函数运算(加法，乘法等，elementary operations)。
然后在 backward 的时候触发每个节点的 gradient 计算，并将计算完成的 gradient 存储在各自对应的<code style="color: #B58900">.grad</code>属性内。
最后 optim 的 step 执行时将每个参数的值使用对应<code style="color: #B58900">.grad</code>属性内的 gradient 进行更新计算：<code style="color: #B58900">p = p - lr * p.grad</code>。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">如何构建计算图？首先，torch 构造的计算图是一个有向无环图(DAG)。
其中，叶子节点为输入的 tensor，包括需要计算梯度的输入的数据和参与计算的模型参数，根节点为输出的 tensor，而中间节点是模型执行的每个初等函数运算(简称为执行操作)。
假设<code style="color: #B58900">a</code>和<code style="color: #B58900">b</code>是<code style="color: #B58900">torch.Tensor</code>变量，
且<code style="color: #B58900">M = lambda x, y: 3*x**3 - y**2</code>，则在<code style="color: #B58900">out=M(a,b)</code>时，<code style="color: #B58900">torch.autograd</code>构造了如下的 DAG，
其中每一个节点表示一个初等函数：</p>

![DAG](/images/torch_autograd_DAG.png)

<p style="text-align:justify; text-justify:inter-ideograph;">其中，<span style="color: blue">蓝色</span>节点表示执行操作节点，在<code style="color: #B58900">torch.autograd</code>中使用<code style="color: #B58900">Function</code>类来实现。
每个初等函数都实现了一个<code style="color: #B58900">Function</code>子类，例如幂函数为<code style="color: #B58900">PowBackward0</code>类。
在<code style="color: #B58900">Function</code>类中，需要实现<code style="color: #B58900">forward</code>和<code style="color: #B58900">backward</code>函数，
其中前者在模型前向运算时使用，而后者在 loss 后向运算时使用。以下为指数函数的<code style="color: #B58900">Function</code>类简易实现：</p>

![exp Function](/images/torch_autograd_Function.png)

<p style="text-align:justify; text-justify:inter-ideograph;">而且每个执行操作<code style="color: #B58900">Function</code>实例都保存在其输出 tensor 的<code style="color: #B58900">.grad_fn</code>属性上。
因此，PyTorch 中的模型训练范式为: 在 forward 时，输入数据和模型参数，对于每一个执行操作，构建一个对应的<code style="color: #B58900">Function</code>实例，
并调用<code style="color: #B58900">.apply()</code>输出结果作为下一个执行操作的输入数据，并将<code style="color: #B58900">Function</code>实例保存在输出结果的<code style="color: #B58900">.grad_fn</code>属性上。
而在 backward 时，首先使用<code style="color: #B58900">l.grad_fn</code>确定输出节点的<code style="color: #B58900">Function</code>实例，
然后调用其<code style="color: #B58900">.backward()</code>计算对于输入数据的 gradient，并将 gradient 结果传递给下一个节点，
即前一个节点的输入数据所对应的<code style="color: #B58900">Function</code>实例。重复上述操作，直到达到 DAG 的叶子节点，表明 gradient 已计算到输入数据/模型参数。
此时，将最终计算得到的 gradient 保存到其<code style="color: #B58900">.grad</code>属性中。
可以看到，默认情况下，PyTorch 不保存中间计算结果的 gradient (即中间结果的<code style="color: #B58900">.grad</code>属性为<code style="color: #B58900">None</code>)。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">此外，PyTorch 的求导是通过计算雅可比矩阵(Jacobian matrix) $J$ 和向量 $\vec{v}$ 的乘积来实现链式法则的反向传播(back propagate)，即：</p>

$$J=\left(\begin{array}{ccc}\frac{\partial \mathbf{y}}{\partial x_1} & \cdots & \frac{\partial \mathbf{y}}{\partial x_n}\end{array}\right)=\left(\begin{array}{ccc}\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n}\end{array}\right), \vec{v}=\left(\begin{array}{ccc}
\frac{\partial l}{\partial y_1} & \cdots & \frac{\partial l}{\partial y_m}
\end{array}\right)^T \\ J^T \cdot \vec{v}=\left(\begin{array}{ccc}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_1} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_1}{\partial x_n} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{array}\right)\left(\begin{array}{c}
\frac{\partial l}{\partial y_1} \\
\vdots \\
\frac{\partial l}{\partial y_m}
\end{array}\right)=\left(\begin{array}{c}
\frac{\partial l}{\partial x_1} \\
\vdots \\
\frac{\partial l}{\partial x_n}
\end{array}\right)$$

<p style="text-align:justify; text-justify:inter-ideograph;">因此<code style="color: #B58900">.backward()</code>理论上只能对数求导，不能对向量求导，
即 $l.backward()$ 中 $l$ 理论上只能是一个数 (即标量)。为了实现向量 $Q$ 的求导，需要添加与 $Q$ 形状相同的初始梯度<code style="color: #B58900">Q.backward(gradient = init_gradient)</code>。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">同时，PyTorch 的计算图是在每次 forward 时构建的，在 backward 后销毁(只是弃用)；然后在下一次 forward 时再次构建。
这样可以保证每次 forward 时的计算图都是最新的，使得可以针对模型进行任意的改动(比如训练前半段更新全部参数，训练后半段更新指定参数)。</p>

Computational Graph Implementation
===

<p style="text-align:justify; text-justify:inter-ideograph;">下面的代码简单地实现了<code style="color: #B58900">torch.autograd</code>关于模型训练的过程，包括前向构建计算图并计算出模型输出；利用计算图反向传播并存储梯度。</p>

![implement](/images/torch_autograd_implement.png)

<p style="text-align:justify; text-justify:inter-ideograph;">下图展示了一个<code style="color: #B58900">torch.autograd</code>构建计算图的实例，其中<span style="color: #B58900">黄色</span>节点表示无需计算梯度的输入数据；
<span style="color: green">绿色</span>节点表示计算图的叶子节点对应的数据；<span style="color: saddlebrown">褐色</span>节点表示中间计算节点所对应的数据；
<span style="color: blue">蓝色</span>节点表示<code style="color: #B58900">torch.autograd</code>构建的计算图：</p>

![DAG2](/images/torch_autograd_DAG2.png)

References
===

1. [Pytorch Autograd In-Deep Introduction Video](https://www.youtube.com/watch?v=MswxJw-8PvE)
2. [A Gentle Introduction to torch.autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html?highlight=torch%20autograd)
3. [Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
4. [Pytorch Autograd API Documentation](https://pytorch.org/docs/stable/autograd.html)