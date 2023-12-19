---
title: 'The Basic Knowledge of Gradient Penalty'
date: 23-12-19
permalink: /posts/2023/12/blog-gradient-penalty/
star: superior
tags:
  - 深度学习基本知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要讲解了使用梯度惩罚(gradient penalty)作为正则化项来促进模型学习的数学原理和具体实现。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">我们知道，最常见的关于参数的正则化是将参数 $\mathcal{W}$ 的 $L_1/L_2$ 作为正则化项加入到 loss 中以限制模型的复杂性。
在 PyTorch 中可以通过设置优化器中的<code style="color: #B58900">weight_decay</code>参数来实现。
本文介绍的梯度惩罚与其稍有不同，它是利用梯度作为正则化项来促进模型的学习以获得更好的性能(但是不能保证模型的简单性)。
它主要包括 $2$ 中方式：<b>数据梯度惩罚</b>和<b>参数梯度惩罚</b>。下面将详细介绍：</p>

<p style="text-align:justify; text-justify:inter-ideograph;">数据梯度惩罚：在深度学习中，存在一种”对抗样本“，它仅仅是在原始样本的基础上添加一些我们难以察觉的随机噪声，便可使模型的结果错误。
典型例子如下：</p>

![adversarial example](/images/adversarial_example.png)

<p style="text-align:justify; text-justify:inter-ideograph;">为了缓解这种情况，就需要构造”对抗样本“给模型学习，来增强模型的鲁棒性，即<b>对抗训练</b>。
具体而言，假设原始样本为 $x$，”对抗样本“为 $x+\Delta x$，其中，$\Delta x$ 是随机噪声，
其要求是范围有限(不然就不能和原始样本看起来相似)，并且需要使模型的结果越差越好。
在具体实现时，限制 $\Delta x$ 的范围可以使用限制其的 $L_1$ 范数：$||\Delta x|| \leq \epsilon$，其中 $\epsilon$ 是一个很小的正数；
而衡量模型的结果可以使用损失函数 $\mathcal{L}(x,y;\theta)$，其中 $y$ 表示数据标签，$\theta$ 表示模型参数。
要想加入 $\Delta x$ 后模型结果最差，即使得损失函数值最高：$\underset{\Delta x \in \Omega}{max}{\mathcal{L}(x+\Delta x,y;\theta)}$。
通过求解满足这 $2$ 个条件的 $\Delta x$，便可获得原始样本为 $x$ 的”对抗样本“ $x+\Delta x$。
然后将其送入模型进行训练，使得模型对其标签 $y$ 预测尽可能正确，即使得损失函数最小：$\underset{\theta}{min}{\mathcal{L}(x+\Delta x,y;\theta)}$。
最后通过对每个样本都构造”对抗样本“进行学习，即可完成对抗训练。因此，最终对抗训练的表达式如下：</p>

$$\underset{\theta}{min}\mathbb{E}_{(x,y) \sim \mathcal{D}}\underset{\Delta x \in \Omega; ||\Delta x|| \leq \epsilon}{max}{\mathcal{L}(x+\Delta x,y;\theta)}$$

<p style="text-align:justify; text-justify:inter-ideograph;">其中 $\mathcal{D}$ 表示训练数据集。那么如何计算 $\Delta x$？其目的是增大损失函数 $\mathcal{L}(x,y;\theta)$，即梯度增加的方向，
因此可以简单地取 $\Delta x = \triangledown_x \mathcal{L}(x,y;\theta)$。注意：这里的梯度是关于数据 $x$ 的梯度，而不是关于参数 $\theta$ 的梯度。
同时为了限制 $\Delta x$ 的范围，需要对其归一化处理，最终可得：</p>

$$\Delta x = \epsilon \dfrac{\triangledown_x \mathcal{L}(x,y;\theta)}{||\triangledown_x \mathcal{L}(x,y;\theta)||}$$

<p style="text-align:justify; text-justify:inter-ideograph;">因此，对抗训练的最终算法如下：</p>

1. <p style="text-align:justify; text-justify:inter-ideograph;">先使用数据集 $\mathcal{D}$ 训练一个初始模型 $\mathbf{M}_\theta: \underset{\theta}{min}{\mathcal{L}(x,y;\theta)}$;
2. <p style="text-align:justify; text-justify:inter-ideograph;">然后对于数据集 $\mathcal{D}$ 中的每个样本 $\{x_i, y_i\}$，构造”对抗样本“  $\{x_i+\Delta x, y_i\} = \{x_i + \epsilon \dfrac{\triangledown_x \mathcal{L}(x,y;\theta)}{||\triangledown_x \mathcal{L}(x,y;\theta)||}, y_i\}$;
3. <p style="text-align:justify; text-justify:inter-ideograph;">接着使用构造的”对抗样本“继续训练模型 $\mathbf{M}_\theta = \underset{\theta}{min}\mathbb{E}_{(x,y) \sim \mathcal{D}}{\mathcal{L}(x+\Delta x,y;\theta)}$;
4. <p style="text-align:justify; text-justify:inter-ideograph;">循环 $2,3$ 步直至收敛。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">回看步骤 $3$ 的对抗训练损失：$\mathcal{D}{\mathcal{L}(x+\Delta x,y;\theta)}$。
由于 $\Delta x$ 很小，将其进行<b>泰勒一阶展开</b>可得：</p>

$$\begin{align}\mathcal{D}{\mathcal{L}(x+\Delta x,y;\theta)} & = \mathcal{D}{\mathcal{L}(x,y;\theta)}+\triangledown_x\mathcal{L}(x,y;\theta)\Delta x \\
& = \mathcal{D}{\mathcal{L}(x,y;\theta)}+\triangledown_x\mathcal{L}(x,y;\theta) \times \epsilon \dfrac{\triangledown_x \mathcal{L}(x,y;\theta)}{||\triangledown_x \mathcal{L}(x,y;\theta)||} \leftarrow \color{green}{\Delta x = \epsilon \dfrac{\triangledown_x \mathcal{L}(x,y;\theta)}{||\triangledown_x \mathcal{L}(x,y;\theta)||}}\end{align}$$