---
title: 'Stable Diffusion'
date: 23-10-15
permalink: /posts/2023/10/blog-paper-stablediffusion/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html" target="_blank" title="Stable Diffusion">High-Resolution Image Synthesis With Latent Diffusion Models</a></p>

发表会议：The IEEE / CVF Computer Vision and Pattern Recognition Conference(CVPR 2022)

第一作者：Robin Rombach(LMU(Ludwig Maximilian University) Munich)

Question
===
<p style="text-align:justify; text-justify:inter-ideograph;"> 如何减少 DM(Diffusion Model) 的 train 计算量和提高 inference 速度，同时保证其生成图像的逼真性，并添加额外的 condition 控制模型生成。 </p>

Preliminary
===

![Diffusion Model](/images/paper_ControlNet_Diffusion_Model.jpg)

<p style="text-align:justify; text-justify:inter-ideograph;"> Diffusion Model：扩散模型是近年来最热的图像生成思想。
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

<p style="text-align:justify; text-justify:inter-ideograph;"> DM 虽然在图像逼真度上已经超过了目前的 SOTA(GAN)　</p>