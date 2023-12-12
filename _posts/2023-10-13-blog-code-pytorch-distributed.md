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

<p style="text-align:justify; text-justify:inter-ideograph;">由上述单 GPU 训练过程可以看到，该训练范式存在多个可以进行划分的维度：</p>

1. <p style="text-align:justify; text-justify:inter-ideograph;">将 $B$ 进行划分($B = [B_1,...,B_M]$)，然后将不同的输入数据 $X_m \in \mathbb{R}^{B_m \times N}$ 和 $label_m \in \mathbb{R}^{B_m \times 1}, m = [1,...,M]$ 分布到不同的 GPU 进行计算，即可得到 Data Parallelism；</p>
2. <p style="text-align:justify; text-justify:inter-ideograph;">将 $N$ 进行划分($N = [N_1,...,N_H]$)，然后将不同的输入数据 $X_h \in \mathbb{R}^{B \times N_h}$ 和 $label_h \in \mathbb{R}^{B \times 1}, h = [1,...,H]$ 分布到不同的 GPU 进行计算；即可得到 Sequence Parallelism；</p>
3. <p style="text-align:justify; text-justify:inter-ideograph;">将 $L$ 进行划分($L = [L_1,...,L_I]$)，然后将不同的模型模块 $\mathbf{M}_i \in \mathbb{R}^{L_i \times D}, i=[1,...,I]$ 分布到不同的 GPU 进行计算，即可得到 Parallelism Parallelism；</p>
4. <p style="text-align:justify; text-justify:inter-ideograph;">将 $D$ 进行划分($D = [D_1,...,D_J]$)，然后将不同的模型参数 $\mathbf{M}_j \in \mathbb{R}^{L \times D_j}, j=[1,...,J]$ 分布到不同的 GPU 进行计算，即可得到 Tensor Parallelism。</p>


Data Parallelism (DP) $\rightarrow$ Distributed Data Parallelism (DDP)
===

<p style="text-align:justify; text-justify:inter-ideograph;">通俗地讲，DP 将相同的模型参数复制到多个 GPU 上(通常称为 “worker”)，并为每个 GPU 分配不同的数据(即 $\{X_m, label_m\}$)以实现同时处理。
具体而言，假设有 $M$ 个 GPU 可供训练，DP 首先在每个 GPU 上都存储一份模型参数 $\mathbf{M}$，然后为第 $m$ 个 GPU 分配第 $m$ 份训练数据 $\{X_m, label_m\}$。
接着执行下面操作：</p>

1. 独立计算每个 worker 关于对应数据 $\{X_m, label_m\}$ 的梯度 $\mathcal{G}_m$；
2. 对所有 worker 的梯度进行收集并平均：$\mathcal{G} = \dfrac{\sum_{m=1}^M{\mathcal{G}_m}}{M}$；
3. 在每个 worker 上独立使用 optimizer 计算新参数：$\hat{\mathbf{M}} = \mathbf{M} - \eta \times \mathcal{G}$。

<p style="text-align:justify; text-justify:inter-ideograph;">因此，DP 本身仍然要求设计的模型适合单个 GPU 的内存(也就是在 batch_size$= 1$ 的情况下，模型必须能在单 GPU 的训练范式下成功训练，而不会爆 OOM)，
只是允许利用多个 GPU 的计算能力来在单位时间内处理更多的训练数据(即处理更大的 $B$)，且其代价是存储许多重复的参数副本(每个 GPU 都存储一份)。
也就是说，有一些策略可以增加 GPU 可用的有效 RAM，例如在使用之间临时将参数卸载到 CPU 内存。
同时，当每个 worker 更新其参数副本时，它们需要协调以确保每个 worker 在更新后仍然拥有相同的参数，即更新后的 $\hat{\mathbf{M}}$ 也应该相同。
最简单的方法是在 worker 的各个阶段引入阻塞式通信(block communication)。可以看到，步骤 $2$ 是一个求梯度平均值的操作，可以在这里进行阻塞通信，
即等到所有的 worker 都计算完成各自的梯度 $\mathcal{G}_m$，然后在进行通信获得平均的梯度 $\mathcal{G}$。
也就是说，对于任意一个 worker，在它计算完成自己的梯度之后，只能等待其他 worker 计算完成它们的梯度，并得到最终的均值梯度后，才能使用自己的 optimizer 进行参数更新并进行下一轮迭代。
这极大地阻碍了 GPU 计算资源的利用(因为在等待期间 worker 什么也没干，即空闲时间)。因此 <b>PyTorch</b> 实现更为复杂的通信方法来尽可能减少 worker 的空闲时间。</p>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>
