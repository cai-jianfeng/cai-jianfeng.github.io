---
title: 'Emu series (Emu & Emu Edit & Emu Video)'
date: 23-11-21
permalink: /posts/2023/11/blog-paper-emu-series/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;">本文主要对近期 Meta 发表的三篇关于视觉处理的文章(Emu 系列)进行论文解读(按照它们的发布顺序)：
首先是 SOTA 的 text-to-image 生成模型 Emu；接着以它为 baseline，进行 image edit 的研究改进，提出了一个大一统的图像编辑模型 Emu Edit，
这基本上就把图像领域主流的任务都刷了个遍。最后又提出了 Emu Video 模型，利用 Emu 完成了对 text-to-video 生成模型的改进，也获得了 SOTA。
(ps：我猜下一步应该就是 video edit 的研究改进了🙂)</p>

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://arxiv.org/abs/2309.15807" target="_blank" title="Emu">Emu: Enhancing Image Generation Models Using Photogenic Needles in a Haystack</a></p>

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://arxiv.org/abs/2311.10089" target="_blank" title="Emu Edit">Emu Edit: Precise Image Editing via Recognition and Generation Tasks</a></p>

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://arxiv.org/abs/2311.10709" target="_blank" title="Emu Video">Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning</a></p>

<p style="text-align:justify; text-justify:inter-ideograph;">发表会议：三篇论文都发表在 Conference of Computer Vision and Pattern Recognition (CVPR 2023)</p>

<p style="text-align:justify; text-justify:inter-ideograph;">第一作者：Xiaoliang Dai & Shelly Sheynin & Rohit Girdhar (GenAI, Meta)</p>

Preliminary
===

<p style="text-align:justify; text-justify:inter-ideograph;">Diffusion Model (DM)：扩散模型是近年来最热的图像生成思想。
将任意一张图像 $I_0$ 进行噪声添加，每次添加一个服从 $N(0,1)$ 分布的随机噪声 $\epsilon_t$，获得含有噪声的图像 $I_t$，则进行了无数次后，原本的图像就会变成一个各向同性的随机高斯噪声。
按照这个理论，我们对一个各向同性的随机高斯噪声每次添加一个特定的服从 $N(0,1)$ 分布的噪声 $\epsilon_t'$，获得噪声量减少的图像 $I_t'$，则经过足够多次后便可获得一张逼真的图像 $I_0'$。
为此，我们可以选择任意一个图像生成模型 $M_G$ (例如 U-net 等)，第 t 次时输入第 t-1 次生成的图像 $I_{t-1}'$，输出生成的图像 $I_t'$。
由于每次使用的是同一个图像生成模型 $M_G$，所以需要输入时间步信息 $t$ 来告诉模型现在是生成的第几步
(一个直观的理解是在随机噪声恢复成原始图像的前期，模型更关注图像的整体恢复，而后期则更加关注图像的细节恢复，所以需要时间步信息告诉模型现在是偏前期还是后期)。
同时通过 $I_{t-1}$ 直接生成 $I_t$ 较为困难，因为它需要生成正确其每个像素值，而每次添加的噪声 $\epsilon_t$ 较容易预测。
因此可以更换为每次输入 $I_{t-1}$，模型预测第 $t$ 次所加的噪声 $\epsilon_t$。所以损失函数为 $L = E_{z_0, t, c_t, \epsilon \sim N(0,1)}[||\epsilon - \epsilon_{\theta}(z_t, t, c_t)||_2^2]$。
其中 $z_0$ 是原始噪声(即一开始输入的图像)，而 $z_t$ 是第 $t$ 步输出的图像, $\epsilon_{\theta}(z_t, t, c_t)$ 为模型预测的第 $t$ 步所加的噪声。
更加详细的介绍推理可以参考 <a href="https://cai-jianfeng.github.io/posts/2023/11/blog-diffusion-model/" target="_blank">The Basic Knowledge of Diffusion Model (DM)</a>或者 <a href="https://cai-jianfeng.github.io/posts/2023/11/blog-score-based-generative-model/" target="_blank">The Basic Knowledge of Scored-based Generative Model</a>。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">Stable Diffusion：是 DM 模型的一种改进形式。它将噪声推广到潜变量空间进行学习，使得模型的计算量大大降低。
同时通过 cross-attention 的方法添条件，使得模型可以根据给定的条件(如 text)来生成指定的图像。
详细的介绍推理可以参考 <a href="https://cai-jianfeng.github.io/posts/2023/10/blog-paper-stablediffusion/" target="_blank">Stable Diffusion</a>。</p>

<h1>Emu</h1>

<h2>Question</h2>

<p style="text-align:justify; text-justify:inter-ideograph;">如何使得 Diffusion Model (DM) 可以生成高质量的图像，同时又保持其泛化性(即能够生成任意描述的图像)</p>

<h2>Method</h2>

<p style="text-align:justify; text-justify:inter-ideograph;">想要 DM 模型生成任意(即大范围)的高质量图像，最简单直观的方法是收集一个这样的数据集给模型训练，但这显然是不可能的。
因此，一般采用的方法是先在一个质量参差不齐的大规模图像数据集(通常是网上收集到的)上预训练一个 DM 模型(称为 pre-training)。
此时它具有了生成任意的图像的能力，即<b>泛化能力</b>，但是生成质量不够高。接着便设计一个方法(称为 post pre-training)，进一步改进模型的<b>生成质量</b>，同时又能保持模型的泛化能力。
本文便提出了一个简单好用的 post pre-training 方法(称为 <b>quality-tuning</b>)来改进模型。
具体而言，本文首先使用 latent Diffusion Architecture (即 <a href="https://cai-jianfeng.github.io/posts/2023/10/blog-paper-stablediffusion/" target="_blank">Stable Diffusion</a>)作为生成模型，
并对其进行简单改进以增强其作为预训练模型的能力(即在没有 post pre-training 之前，就尽量将模型的性能提高。
因为本文发现，对于提出的 quality-tuning 方法而言，如果原先的预训练模型能力越强，这经过 quality-tuning 后的模型能力也会越强)。
模型的具体改动如下：</p>

<ul><li><p style="text-align:justify; text-justify:inter-ideograph;">将 autoencoder 的 channel 数量从 4 提高到 16。</p></li>
<li><p style="text-align:justify; text-justify:inter-ideograph;">添加额外的 adversarial loss 进行训练。</p></li>
<li><p style="text-align:justify; text-justify:inter-ideograph;">将原始的 RGB 图像输入使用傅里叶特征变换将其变换到更高 channel 维度的输入。</li>
<li><p style="text-align:justify; text-justify:inter-ideograph;">增加 U-Net 模型的 channel 数量和 residual block 的数量。</li>
<li><p style="text-align:justify; text-justify:inter-ideograph;">使用 CLIP ViT-L 将图像转为 visual embedding；使用 T5-XXL 将文本转为 text embedding (text 是作为条件)。</li></ul>

<h1>Emu Edit</h1>

<h1>Emu Video</h1>

