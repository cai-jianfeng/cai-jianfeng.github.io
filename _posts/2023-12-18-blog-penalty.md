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

