---
title: 'ControlNet'
date: 23-10-13
permalink: /posts/2023/10/blog-paper-controlnet/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html" target="_blank" title="ControlNet">Adding Conditional Control to Text-to-Image Diffusion Models</a></p>

发表会议：International Conference on Computer Vision(ICCV 2023, Best Paper(Marr Prize))

第一作者：Lvmin Zhang(PhD with Stanford University)

Question
===
<p style="text-align:justify; text-justify:inter-ideograph;"> 如何在 Text-to-Image Diffusion 图像生成模型中添加条件控制其图像的生成(这里的条件主要包括 visual condition，如 mask, depth等，即输入的条件也是图像)，同时保证其生成图像的逼真性。 </p>

Preliminary
===
<p style="text-align:justify; text-justify:inter-ideograph;"> Diffusion Model：扩散模型 </p>

Method
===

![ControlNet Architecture](/images/paper_ControlNet.png)

<p style="text-align:justify; text-justify:inter-ideograph;"> 这个问题最简单的方法是找到一个带有该条件(假设为 $C_i$)的数据集，再在别人已经预训练好的 Difussion 模型上进行微调。
但是这么做有一个问题：因为预训练的 Diffusion Model 是在大量图像上训练而来，如果直接将整个模型直接进行有监督微调，可能会使得模型生成效果大打折扣(即文中说的 overfitting 和 catastrophic forgetting)。
所以本文解决的是在加入了条件 $C_i$ 的情况下，既能输出符合条件 $C_i$ 的图像，又能保证其逼真度；同时，本文还解决了多个 $C_i$ 同时作用于一张图像的生成问题，并且训练的参数也较少。
本文的方法较为简单，可以看作是一个即插即用的插件模块。如图 1，对于任意一个 Diffusion 模型，它通常是由每个基本单元块 $net_{b}$ 组成(如 resnet block, transformer block等)。
原始的 Diffusion 模型输入上一次 $net_{b}$ 生成的 feature map $x$，输出更新的 feature map $y$。
而 ControlNet 保持每个 $net_b$ 的参数不动(即 freeze parameter)，并复制出一个相同的 $net_b'$，称为 $trainable\ copy$，作为条件 $C_i$ 的处理模块。
并在 $trainable\ copy$ 都添加了额外的模块 $zero\ convolution$：它是一个卷积层，其卷积核权重和偏置都初始化为 0。
设 $net_b$ 的数学函数为 $\digamma(x;\Theta)$，$zero\ convolution$ 的数学函数为 $Z(·;·)$，
则 ControlNet 的输入输出公式为 $y_c = \digamma(x;\Theta) + Z(\digamma(x + Z(c;\Theta_z1);\Theta_c); \Theta_{z2})$。
由于 $zero\ convolution$ 的参数初始化为 0，所以 $Z(c;\Theta_z1) = 0$，
$\digamma(x + Z(c;\Theta_z1);\Theta_c) = \digamma(x;\Theta_c)$，则 $trainable\ copy$ 在开始时依旧接收 $x$ 作为输入，和无条件训练的情况一致。
这样可以极大程度保持原始预训练模型的 capability。
同时 $Z(\digamma(x + Z(c;\Theta_z1);\Theta_c); \Theta_{z2}) = 0$，
所以 $y_c = \digamma(x;\Theta) + Z(\digamma(x + Z(c;\Theta_z1);\Theta_c); \Theta_{z2}) = \digamma(x;\Theta)$。
这样，在训练刚开始时，有害噪声就不会影响 $trainable\ copy$ 中神经网络层的隐藏状态。
而训练损失函数则和普通的 Diffusion Model 类似：假设 time step 为 $t$(Diffusion 模型训练的时间步), 
text prompts 为 $c_t$(文本条件), visual condition 为 $c_f$(即我们自己添加的额外条件)，训练模型为 $\epsilon_{theta}$，
则损失函数为 $L = E_{z_0, t, c_t, c_f, \epsilon \sim N(0,1)}[||\epsilon - \epsilon_{theta}(z_t, t, c_t, c_f)||_2^2]$</p>