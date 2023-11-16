---
title: 'The Basic Knowledge of Scored-based Generative Model'
date: 23-11-16
permalink: /posts/2023/11/blog-score-based-generative-model/
tags:
  - 深度学习基础知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客参考了<a href="https://yang-song.net/blog/2021/score/#connection-to-diffusion-models-and-others" target="_blank">
Generative Modeling by Estimating Gradients of the Data Distribution</a>，
详细讲述了最近大火的 Diffusion Model 的另一个理解/推理角度: Score-based Generative Model 的数学原理及编程。</p>

Score-Based Generative Model 的基本原理
===

<p style="text-align:justify; text-justify:inter-ideograph;">给定一个数据集 $D = \{x_1,...,x_N\}, x_i \sim p(x)$，
生成模型(generative model) $M_\theta(·)$ 的目标是要拟合 $p(x)$，使得可以通过采样 $M_\theta(·)$ 来生成新的数据 $\hat{x}$。
最直接简单的方式是训练一个模型 $M_\theta(x)$ 拟合数据集 $D$ 的概率分布函数(p.d.f.) $p(x)$，
即 $\boldsymbol{M_\theta(x) = p_\theta(x) = \dfrac{e^{-f_\theta(x)}}{Z_\theta} \approx p(x),Z_\theta > 0, \int p_\theta(x)dx = 1}$。
其中，$\boldsymbol{f_\theta(x)}$ 通常被称为 unnormalized probabilistic model/energy-based model。
然后使用常用的训练方式 —— maximize log-likehood (最大化对数似然)来训练模型 $M_\theta(x) = p_\theta(x)$：$L_\theta = \underset{\theta}{max}\sum_{i=1}^{N}{log\ \boldsymbol{p_\theta(x_i)}}$。
这样便可以训练一个生成模型来生成数据。但是这种做法的最大局限性在于必须满足 $\int p_\theta(x)dx = 1$，这就使得必须计算出 normalizing constant $Z_\theta = \int e^{-f_\theta(x)}dx$，
对于任何一个 general $f_\theta(x)$ 来说这都是一个经典的难以计算的量。
因此常见的方法要么限制模型的架构使得 $Z_\theta$ 可以计算(限制了模型的灵活性)，要么使用其他方式近似 $Z_\theta$ (存在偏差)，它们都存在一定的问题。
但是如果我们能找到一个替代函数，使得函数中仅包含 $f_\theta{x}$，那么就可以避免计算 $Z_\theta$，这样就可以既保持模型架构的灵活性，又不引入偏差。
本文便找到了一个简单的函数 $\triangledown_xlog\ p(x)$，称为 $p(x)$ 的 <b>score function</b> 可以作为替代函数。其证明推导如下：</p>

<center>$\triangledown_xlog\ p_\theta(x) = \triangledown_xlog \dfrac{e^{-f_\theta(x)}}{Z_\theta} = -\triangledown_xf_\theta(x) -\underset{=0}{\underbrace{ \triangledown_xlog\ Z_\theta}} = -\triangledown_xf_\theta(x)$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">可以看到，$\triangledown_xlog\ p(x)$ 中没有包含 $Z_\theta$。那么我们只需要训练一个模型 $s_\theta(x)$ 拟合 $\triangledown_xlog\ p(x)$ 就可以了，
即 $\boldsymbol{s_\theta(x) \approx \triangledown_xlog\ p(x)}$，其中的拟合模型 $s_\theta(x)$ 称为 score-based model。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">而在训练方面，由于 $\triangledown_xlog\ p(x)$ 仍然是一个密度分布，则可以使用常见的 minimize <b>**Fisher divergence</b> 来训练：</p>

<center>$\mathbb{E}_{p(x)}[||\triangledown_xlog\ p(x) - s_\theta(x)||_2^2]$</center>

