---
title: 'Consistency Model'
date: 23-11-14
permalink: /posts/2023/11/blog-paper-consistency-model/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://openreview.net/forum?id=FmqFfMTNnv" target="_blank" title="Consistency Model">Consistency Models</a></p>

<p style="text-align:justify; text-justify:inter-ideograph;">发表会议：International Conference on Machine Learning (ICML 2023)</p>

第一作者：Yang Song (OpenAI)

Question
===

<p style="text-align:justify; text-justify:inter-ideograph;">如何提高图像生成模型的生成速度，同时尽可能保持高性能(逼真)</p>

Preliminary
===

<p style="text-align:justify; text-justify:inter-ideograph;">Diffusion Model (DM)：如下图(Figure 1)，扩散模型是近年来最热的图像生成思想。
将任意一张图像 $I_0$ 进行噪声添加，每次添加一个服从 $N(0,1)$ 分布的随机噪声 $\epsilon_t$，获得含有噪声的图像 $I_t$，则进行了无数次后，原本的图像就会变成一个各向同性的随机高斯噪声。
按照这个理论，我们对一个各向同性的随机高斯噪声每次添加一个特定的服从 $N(0,1)$ 分布的噪声 $\epsilon_t'$，获得噪声量减少的图像 $I_t'$，则经过足够多次后便可获得一张逼真的图像 $I_0'$。
为此，我们可以选择任意一个图像生成模型 $M_G$ (例如 U-net 等)，第 t 次时输入第 t-1 次生成的图像 $I_{t-1}'$，输出生成的图像 $I_t'$。
由于每次使用的是同一个图像生成模型 $M_G$，所以需要输入时间步信息 $t$ 来告诉模型现在是生成的第几步
(一个直观的理解是在随机噪声恢复成原始图像的前期，模型更关注图像的整体恢复，而后期则更加关注图像的细节恢复，所以需要时间步信息告诉模型现在是偏前期还是后期)。
同时通过 $I_{t-1}$ 直接生成 $I_t$ 较为困难，因为它需要生成正确其每个像素值，而每次添加的噪声 $\epsilon_t$ 较容易预测。
因此可以更换为每次输入 $I_{t-1}$，模型预测第 $t$ 次所加的噪声 $\epsilon_t$。所以损失函数为 $L = E_{z_0, t, c_t, \epsilon \sim N(0,1)}[||\epsilon - \epsilon_{\theta}(z_t, t, c_t)||_2^2]$。
其中 $z_0$ 是原始噪声(即一开始输入的图像)，而 $z_t$ 是第 $t$ 步输出的图像, $\epsilon_{\theta}(z_t, t, c_t)$ 为模型预测的第 $t$ 步所加的噪声。
更加详细的介绍推理可以参考 <a href="https://cai-jianfeng.github.io/posts/2023/11/blog-diffusion-model/">The Basic Knowledge of Diffusion Model (DM)</a>。</p>

Method
===

![Consistency Model](/images/paper_Consistency_Model.png)

<p style="text-align:justify; text-justify:inter-ideograph;">目前 DM 模型最大的不足还是推理的速度太慢，由于需要进行多次逆扩散过程的迭代(即 multi-inference)，导致其推理时间较长。
本文便想构造方法实现一次 inference 直接生成图像。
最直观的方式是设计一个模型(如 U-net) $M_\theta$，输入给定的初始化噪声图像 $x_T$，输出预测的原始图像 $\hat{x}_0$，并使用 ground-truth $x_0$ 作为监督信号，
利用 MSE 函数计算损失 $L_{\theta} = \mathbf{E}[||\hat{x}_0 - x_0||_2^2]$ 进行训练。
当然，也可以和 <a href="https://cai-jianfeng.github.io/posts/2023/11/blog-diffusion-model/">The Basic Knowledge of Diffusion Model (DM)</a> 中提到的一样，
预测出 $x_T$ 所加的噪声 $\varepsilon_T$ 来获得原始图像 $\hat{x}_0 = \dfrac{1}{\sqrt{\bar{\alpha}_T}}(x_T - \sqrt{1 - \bar{\alpha}_T}\bar{\varepsilon}_T)$。
但是这 $2$ 种方法的最大缺陷是性能不足，生成的图像的逼真度不够。
其中很大一部分原因是因为在没有其他条件的帮助下，直接让模型建模 $\mathcal{p}(x_0|x_T)/\mathcal{p}(\varepsilon_T|x_T)$ 的难度太大。
为此本文提出通过使用 probability flow (PF) ordinary differential equation (ODE) (概率流常微分方程) 的轨迹来帮助模型学习。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">如上图(Figure 1)，</p>

![Comsistency Model Algorithm](/images/paper_Consistency_Model_Algorithm.png)