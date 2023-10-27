---
title: 'MDM'
date: 23-10-27
permalink: /posts/2023/10/blog-paper-mdm/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://arxiv.org/abs/2310.15111" target="_blank" title="MDM">Matryoshka Diffusion Models</a></p>

<p style="text-align:justify; text-justify:inter-ideograph;">截至2023年10月26日暂无，论文版本为 arxiv-v1</p>

第一作者：Jiatao Gu (Apple)

Question
===

<p style="text-align:justify; text-justify:inter-ideograph;">如何使用一个 end-to-end 架构，实现生成高分辨率的逼真图像($1024 \times 1024$)，同时使用较少的计算量和较为简单的优化函数(即损失)。</p>

Preliminary
===

![Diffusion Model](/images/paper_ControlNet_Diffusion_Model.jpg)

<p style="text-align:justify; text-justify:inter-ideograph;"> Diffusion Model (DM)：扩散模型是近年来最热的图像生成思想。
将任意一张图像 $I_0$ 进行噪声添加，每次添加一个服从 $N(0,1)$ 分布的随机噪声 $\epsilon_t$，获得含有噪声的图像 $I_t$，则进行了无数次后，原本的图像就会变成一个各向同性的随机高斯噪声。
按照这个理论，我们对一个各向同性的随机高斯噪声每次添加一个特定的服从 $N(0,1)$ 分布的噪声 $\epsilon_t'$，获得噪声量减少的图像 $I_t'$，则经过足够多次后便可获得一张逼真的图像 $I_0'$。
为此，我们可以选择任意一个图像生成模型 $M_G$ (例如 U-net 等)，第 t 次时输入第 t-1 次生成的图像 $I_{t-1}'$，输出生成的图像 $I_t'$。
由于每次使用的是同一个图像生成模型 $M_G$，所以需要输入时间步信息 $t$ 来告诉模型现在是生成的第几步
(一个直观的理解是在随机噪声恢复成原始图像的前期，模型更关注图像的整体恢复，而后期则更加关注图像的细节恢复，所以需要时间步信息告诉模型现在是偏前期还是后期)。
同时通过 $I_{t-1}$ 直接生成 $I_t$ 较为困难，因为它需要生成正确其每个像素值，而每次添加的噪声 $\epsilon_t$ 较容易预测。
因此可以更换为每次输入 $I_{t-1}$，模型预测第 $t$ 次所加的噪声 $\epsilon_t$。所以损失函数为 $L = E_{z_0, t, c_t, \epsilon \sim N(0,1)}[||\epsilon - \epsilon_{\theta}(z_t, t, c_t)||_2^2]$。
其中 $z_0$ 是原始噪声(即一开始输入的图像)，而 $z_t$ 是第 $t$ 步输出的图像, $\epsilon_{\theta}(z_t, t, c_t)$ 为模型预测的第 $t$ 步所加的噪声。</p>

Method
===

![MDM-architecture](/images/paper_MDM_architecture.png)

<p style="text-align:justify; text-justify:inter-ideograph;">目前 DM 模型想要生成高分辨率的图像主要有 $2$ 种方式：一是先使用 DM 模型生成分辨率较低的图像，再通过超分 (super-resolution) 模型将分辨率较低的图像转化为高分辨率图像。
这就使得至少需要 2 个模型实现高分辨率；同时，由于是分开训练 DM 和超分模型，这就使得模型计算量的增加，同时生成质量受到 exposure bias 的限制。
第二种方法是将图像转化到维度(即分辨率)更低的潜变量空间，然后实现扩散生成，如 <a href="https://cai-jianfeng.github.io/posts/2023/10/blog-paper-stablediffusion/" target="_blank" title="Stable Diffusion">Stable-Diffusion (LDM)</a>。
这种方法不仅增加了模型学习的复杂性(因为不仅要学习图像的潜变量，还要学习潜变量到像素图像之间的转化)，而且不可避免存在压缩损失 (lossy compression)。
通过上述分析，本文提出了 MDM 模型，结合了第一种方法直接在 pixel 空间上训练的好处，同时又结合了第二种方法 end-to-end 的生成过程(其实第二种方法也不能算 end-to-end，它还是需要从潜变量到 pixel 的后处理过程)。
MDM 模型通过融合了多分辨率图像生成的并行训练方式解决了 end-to-end 模型生成高分辨率图像的问题。其模型思想较为直观：
先前的模型通过 DM 直接生成高分辨率图像的效果不行，很可能主要是因为没有辅助的条件帮助模型学习，导致模型从零生成难度较大。如果能像第一种方法一样，给出低分辨率图像作为参考，模型生成的效率和质量应该会更高。
于是，问题便转化为如何将低分辨率图像的生成融合到高分辨率图像的生成中(即使用一个模型生成)，并使得高分辨率图像的生成能借助低分辨率的图像。
通过观察 U-net 架构(DM 模型最常用的架构)可以看到，U-net 模型主要包括 dowm-sample 将图像 $I$ 下采样为潜表示 $I_l$, 
middle-fuse 对 $I_l$ 进行进一步学习生成 $I_l'$ 和 up-sample 对 $I_l'$ 进行上采样生成原始分辨率图像 $\hat{I}$，
而我们知道， dowm-sample, middle-fuse 和 up-sample 都是对图像大小 unaware 的
(例如，对于一个下采样 $x$ 个像素的 down-sample 模块来说，无论输入多大 $h \times w$ 的图像，都是将其变成 $(h-x) \times (w-x)$)。
所以，一个 U-net 模型可以生成任意分辨率的图像，就解决了一个模型生成任意分辨率图像的问题(其实简单归结就是使用 U-net 模型)。
而对于使用低分辨率图像辅助生成高分辨率图像，MDM 采用先生成低分辨率图像，再将已经生成好的低分辨率图像 $I^{low}$ 和原始的高分辨率图像 $I^{high}$ 经过 down-sample 生成的潜表示 $I_l^{high}$ 相结合 $I^{low} + I_l^{high}$，
作为融合了使用低分辨率图像和原始高分辨率图像的更新潜表示 $I_{l-new}^{high}$，然后再使用 middle-fuse 进行进一步融合，生成 $I_{l-new}^{high}'$，最后通过 up-sample 生成更新后的高分辨率图像 $I^{high}'$。</p>


