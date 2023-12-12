---
title: 'pytorch distributed'
date: 23-10-13
permalink: /posts/2023/10/blog-code-pytorch-distributed/
tags:
  - 深度学习基本知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要介绍了 LLM 分布式并行的训练方式，并着重讲解了 PyTorch 代码的实现 DDP 的方式。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">Large Language Model 的训练需要众多 GPU 或其他 AI accumulator 的联合训练(即 GPU 集群)。
通过将不同维度分布到 GPUs 上，可以实现不同的并行训练方式。
具体而言，主要包括 <b>Data Parallelism</b>、<b>Pipeline Parallelism</b>、<b>Tensor Parallelism</b> 和 <b>Expert Parallelism</b>。</p>

![types of parallelism](images/train_parallelism.png)

<p style="text-align:justify; text-justify:inter-ideograph;">首先回顾一下在单个 GPU 上训练模型的范式。
通常而言，主要包括 $3$ 个步骤：</p>

1. 使用输入(inputs)和标签(labels)前向(forward)通过模型，计算损失(loss)；
2. 使用损失后向(backward)通过模型，计算模型每个参数的梯度(gradient)；
3. 使用优化器(optimizer)根据 gradient 对模型参数进行更新。

<p style="text-align:justify; text-justify:inter-ideograph;">具体而言，首先输入 $X \in \mathbb{R}^{B \times N}$ 和 $label \in \mathbb{R}^{B \times 1}$ forward 模型 $\mathbf{M} \in \mathbb{R}^{L \times D}$，
得到损失 $\mathcal{L} = \mathbf{M}(X, label) \in \mathbb{R}^{B \times 1}$。
然后使用 $\mathcal{L}$ backward 模型得到每个参数的梯度 $\mathcal{G} = BP(loss) \in \mathbb{R}^{L \times D}$。
最后使用 optimizer 根据 $\mathcal{G}$ 对模型 $\mathbf{M}$ 参数进行更新：$\hat{\mathbf{M}} = \mathbf{M} - \eta \times \mathcal{G}$。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">有上述过程可以看到，该训练范式存在多个可以进行划分的维度：</p>

1. 将 $b$ 进行划分($B = [B_1,...,B_M]$)，然后将不同的输入数据 $X_m \in \mathbb{R}^{B_m \times N}$ 和 $label_m \in \mathbb{R}^{B_m \times 1}, m = [1,...,M]$ 分布到不同的 GPU 进行计算，
即可得到 Data Parallelism；
2. 将 $L$ 进行划分($L = [L_1,...,L_I]$)，然后将不同的模型模块 $\mathbf{M}_i \in \mathbb{R}^{L_i \times D}, i=[1,...,I]$ 分布到不同的 GPU 进行计算，即可得到 Parallelism Parallelism；
3. 将 $D$ 进行划分($D = [D_1,...,D_J]$)，然后将不同的模型参数 $\mathbf{M}_j \in \mathbb{R}^{L \times D_j}, j=[1,...,J]$ 分布到不同的 GPU 进行计算，即可得到 Tensor Parallelism；
4. 将 $N$ 进行划分($N = [N_1,...,N_H]$)，然后将不同的输入数据 $X_h \in \mathbb{R}^{B \times N_h}$ 和 $label_h \in \mathbb{R}^{B \times 1}, h = [1,...,H]$ 分布到不同的 GPU 进行计算，
即可得到 Sequence Parallelism。

