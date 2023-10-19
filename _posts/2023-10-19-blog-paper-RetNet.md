---
title: 'RetNet'
date: 23-10-19
permalink: /posts/2023/10/blog-paper-retnet/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://arxiv.org/abs/2307.08621" target="_blank" title="RetNet">Retentive Network: A Successor to Transformer for Large Language Models</a></p>

发表会议：截至2023年10月19日暂无，论文版本为 arxiv-v4

第一作者：Yutao Sun(Tsinghua University)

Question
===
![impossible triangle](/images/paper_RetNet_impossible_triangle.png)
<p style="text-align:justify; text-justify:inter-ideograph;"> 如何设计出一种模型架构，能够打破不可能三角的限制，即既要保持 training parallelism, 又要实现 low-cost inference, 同时还要有 strong performance. </p>

Method
===
<p style="text-align:justify; text-justify:inter-ideograph;"> 首先可以明确一点，寻找单一的架构打破不可能三角几乎是无法做到的，所以可行的方法便是通过结合不同架构之间的优势来实现。
而本文通过分析，可以发现传统的 RNN 模型在 inference 时的时间复杂度为 $O(1)$，但是它无法实现并行训练；
而 transformer 可以实现并行训练，但是其 inference 的时间复杂度为 $O(N)$，同时它们都具有 strong performance。
如果能将 RNN 模型和 transformer 模型结合，就可以实现打破不可能三角，但是问题在于如何能将两者结合，使得它们既使用同一套参数，又能实现不同的架构效果。
于是本文便提出了 RetNet，可以在使用相同参数的情况下同时支持三个计算范式：1) parallel; 2) recurrent; 3) chunk-wise recurrent。
<p style="text-align:justify; text-justify:inter-ideograph;"> RetNet 架构和 Transformer 相似，都是由一个个 RetNet block 组成，每个 block 包含一个 multi-scale retention (MSR) 和一个 feed-forward network (FFN).
而 FFN 与 Transformer 一致，因此下面就详细讲解 MSR. 
由于 RetNet 的 parallel (对应 transformer) 和 recurrent (对应 RNN) 使用的是同一套参数，我们就需要数学推导来证明 RNN 可以通过一定的转化变成 transformer.
具体而言，首先给定一个输入序列 $\{x_i\}_{i=1}^{|x|}$，其中 $|x|$ 表示序列的长度。然后经过 word embedding 层得到词嵌入向量：</p>
<center> $X = [x_1, ..., x_{|x|} \in R^{|x| \times d}$ </center>
