---
title: 'PixelDance'
data: 23-11-27
permalink: '/post/2023/11/blog-pixeldance'
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://arxiv.org/abs/2311.10982" target="_blank" title="PixelDance">Make Pixels Dance: High-Dynamic Video Generation</a></p>

<p style="text-align:justify; text-justify:inter-ideograph;">发表会议：截至2023年11月27日暂无，论文版本为 arxiv-v1</p>

<p style="text-align:justify; text-justify:inter-ideograph;">第一作者：Yan Zeng (ByteDance Research)</p>

Question
===

<p style="text-align:justify; text-justify:inter-ideograph;">如何提高 video generation model 的性能，使其生成高动态的视频</p>

Method
===

![PixelDance architecture](/images/paper_PixelDance_model.png)

<p style="text-align:justify; text-justify:inter-ideograph;">之前的方法大都专注于仅通过 text 来生成视频，本文认为仅仅只有 text 条件指导是不够的。
因此，本文提出加上视频的首帧和末帧作为条件来指导模型生成。
具体而言，本文使用 <a href="https://cai-jianfeng.github.io/posts/2023/10/blog-paper-stablediffusion/" target="_blank">Lantent Diffusion Model</a> 作为 baseline，
并在每个 2D 空间卷积层后添加一个 1D 时间卷积层，同时在每个 2D 空间注意力层后添加一个 1D 时间注意力层来将其扩展为 3D 模型 $\mathcal{M}$。
对于 text 条件，本文先使用 CLIP text encoder 将其编码为 text embedding $c^{text}$，
然后使用 cross-attention 层与模型 $\mathcal{M}$ 的 U-net 生成的 hidden-state $h$ 进行交互($h \rightarrpw query, c^{text} \rightarrow key,value$)。
而对于视频的首帧和末帧条件 $\{\mathbf{I}^{first},\mathbf{I}^{last}\}$，
本文先使用模型的 VAE 的 encoder 将其编码为 image embedding $\{\mathbf{f}^{first},\mathbf{f}^{last}\}, \mathbf{f} \in \mathbb{R}^{C \times H \times W}$。
为了不丢失图像的时间位置信息(一个在最开始，一个在最末尾)，本文使用填充将其进行扩展：$c^{image} = [\mathbf{f}^{first}, PADs, \mathbf{f}^{last}] \in \mathbb{R}^{F \times C \times H \times W}$。
然后与原始的噪声视频输入 $z_T \in \mathbb{R}^{T \times C \times W \times H}$ 在 channel 的维度上进行 concat，
形成最终的输入 $\hat{z}_T = concat(c^{image}, z_T) \in \mathbb{R}^{(F+T) \times C \times W \times H}$。</p>