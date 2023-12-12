---
title: 'pytorch distributed'
date: 23-10-13
permalink: /posts/2023/10/blog-code-pytorch-distributed/
tags:
  - 深度学习基本知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要介绍了 LLM 分布式并行的训练方式，并着重讲解了 PyTorch 代码的实现 DDP 的方式。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">Large Language Model 的训练需要众多 GPU 或其他 AI accumulator 的联合训练(即 GPU 集群)。
通过将不同的模块分布到 GPUs 上，可以实现不同的并行训练方式。
具体而言，主要包括 <b>Data Parallelism</b>、<b>Pipeline Parallelism</b>、<b>Tensor Parallelism</b> 和 <b>Expert Parallelism</b>。</p>

![types of parallelism](images/train_parallelism.png)

<p style="text-align:justify; text-justify:inter-ideograph;">首先回顾一下在单个 GPU 上训练模型的范式。
通常而言，主要包括 $3$ 个步骤：</p>

1. 使用输入(inputs)和标签(labels)前向(forward)通过模型，计算损失(loss)；
2. 使用损失后向(backward)通过模型，计算模型每个参数的梯度(gradient)；
3. 使用优化器(optimizer)根据 gradient 对模型参数进行更新。


