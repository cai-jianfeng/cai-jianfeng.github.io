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
因此，本文提出<b>加上视频的首帧和末帧作为条件</b>来指导模型生成。
具体而言，本文使用 <a href="https://cai-jianfeng.github.io/posts/2023/10/blog-paper-stablediffusion/" target="_blank">Lantent Diffusion Model</a> 作为 baseline，
并在每个 2D 空间卷积层后添加一个 1D 时间卷积层，同时在每个 2D 空间注意力层后添加一个 1D 时间注意力层来将其扩展为 3D 模型 $\mathcal{M}$。
对于 text 条件，本文先使用 CLIP text encoder 将其编码为 text embedding $\mathbf{c}^{text}$，
然后使用 cross-attention 层与模型 $\mathcal{M}$ 的 U-net 生成的 hidden-state $h$ 进行交互($h \rightarrpw query, \mathbf{c}^{text} \rightarrow key,value$)。
而对于视频的首帧和末帧条件 $\{\mathbf{I}^{first},\mathbf{I}^{last}\}$，
本文先使用模型的 VAE 的 encoder 将其编码为 image embedding $\{\mathbf{f}^{first},\mathbf{f}^{last}\}, \mathbf{f} \in \mathbb{R}^{C \times H \times W}$。
为了不丢失图像的时间位置信息(一个在最开始，一个在最末尾)，本文使用填充将其进行扩展：$c^{image} = [\mathbf{f}^{first}, PADs, \mathbf{f}^{last}] \in \mathbb{R}^{F \times C \times H \times W}$。
然后与原始的噪声视频输入 $z_T \in \mathbb{R}^{T \times C \times W \times H}$ 在 channel 的维度上进行 concat，
形成最终的输入 $\hat{z}_T = concat(c^{image}, z_T) \in \mathbb{R}^{(F+T) \times C \times W \times H}$。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">在训练时，为了提高模型生成的多样性，本文通过在视频的最后三帧中随机选择一帧作为末帧 $\mathbf{I}^{last}$。
同时，为了提高模型的鲁棒性，也将 image embeddin $c^{image}$ 进行加噪(和 $z_T$ 一致)。
最后，为了提高模型生成的一致性和灵活性，本文使用类似 drop-out 的策略，使用 $\eta$ 的概率将末帧 $\mathbf{I}^{last}$ 替换为对应的 $PADs$ 进行训练(即丢弃末帧条件)。
而在推理时，本文选择在逆扩散过程的前 $\tau$ 步使用末帧条件来指导模型生成具有合理结局的视频，而在后 $\T - tau$ 步不使用末帧条件来促进模型生成合理一致的视频：</p>

$$\tilde{x}_\theta = \begin{cases}\hat{x}_\theta(\mathbf{z}_t, \mathbf{f}^{first}, \mathbf{f}^{last}, \mathbf{c}^{text}), & t < \tau \\
\hat{x}_\theta(\mathbf{z}_t, \mathbf{f}^{first}, \mathbf{c}^{text}), & \tau \leq t \leq T\end{cases}$$

<p style="text-align:justify; text-justify:inter-ideograph;">同时，使用 classifier-free guidance 进一步促进模型生成。</p>