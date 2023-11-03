---
title: 'Transformer'
date: 23-11-03
permalink: /posts/2023/11/blog-paper-transformer/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html" target="_blank" title="Transformer">Attention is All you Need</a></p>

<p style="text-align:justify; text-justify:inter-ideograph;">发表会议：Neural Information Processing Systems (NLPS 2017)</p>

第一作者：Ashish Vaswani (Startup)

Question
===

<p style="text-align:justify; text-justify:inter-ideograph;">如何解决 NLP 任务无法并行训练 sequence 的问题</p>

Preliminary
===

![attention](/images/paper_Transformer_attention.png)

<p style="text-align:justify; text-justify:inter-ideograph;">attention：attention 是将人类的注意力机制引入神经网络的一种方式。人类在进行分类等任务时，更多的是使用比较的方法来进行学习，
即对于自己的需求(即自己掌握的关键特征) $query$，通过将其与每个候选结果 $vector_i$ 的关键特征 $key_i$ 进行比较。
一般而言，两个相似的物体的关键特征也是相似的，即 $query \approx key_{positive}$。这样我们就可以选择到最终的结果。
在数学形式上，如上图(Figure 2 left)，假设当前的 $query = q$，各个候选的结果 $\{vector_i, key_i\} i = [1,...,N]$ 为 $\{k_i, v_i\}$，
则首先我们可以计算 $q$ 和 $k_i$ 之间的相似度来确定其与各个候选结果的相似性(最简单的使用点乘表示相似度)：</p>

<center>$sim(q, k_i) = qk_i^T, i= [1,...,N]$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">然后，我们可以根据每个候选结果的相似性来选择，相似性越高的选的越多，反之选的越少(<b>Rule1</b>)：</p>

<center>$result = \sum_{i=1}^{N}{sim(q,k_i)v_i}, i=[1,...,N]$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">但是这样会产生一个问题，即 $result$ 的数量级 $\approx$ $q$ 的数量级 $+$ $k$ 的数量级 $+$ $v$ 的数量级，
会远远大于候选结果 $v$ 的数量级。因此我们要将相似度进行缩放，将其表示为每个候选结果在 $result$ 中的占比，
这样既保持了 Rule1，又使得 $result$ 和 $v$ 的数量级保持一致。为此，可以使用 $softmax(·)$ 将相似度转化为占比：</p>

<center>$p(q, k_i) = softmax(sim(q,k_i)) = \frac{exp(qk_i^T)}{\sum_{j=1}^N{exp(qk_j^T)}}, i=[1,...,N]$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">最终，$result$ 的表达式为：</p>

<center>$result = \sum_{i=1}^N{p(q,k_i)v_i}, i=[1,...,n]$</center>

Method
===

![Transformer architecture](/images/paper_Transformer_architecture.png)

<p style="text-align:justify; text-justify:inter-ideograph;">传统的 recurrent 模型在序列建模的准确率上已经有了很大的改进，但是其最致命的问题是其训练的顺序性，
导致其训练与推理时长和训练样本的长度成正比，这极大限制了模型可处理的序列长度；而且，只要 recurrent 模型的架构不变，这个问题基本上无法解决(CNN + 隐变量 $h_i$)。
为此，本文放弃了 recurrent 模型的架构，采用了全新的基于 attention 的架构 $Transformer$。
它保留了 Encoder-Decoder 的框架，但是实现了 Encoder 编码序列的并行。
具体而言，如上图，假设输入序列为 $x = (x_1,...,x_n)$，需要将其转化为 $y = (y_1,...,y_m)$。
对于 Encoder，它需要将输入序列 $x$ 转化为之间表示 $z = (z_1,...,z_n)$。它是由一个个 Encoder Block 组成，每个 Encoder Block 的结构相同。
每个 Encoder Block 的主要作用是学习输入序列之间的相互关系，因此需要 attention 来不断关注序列中其他元素与自身的关系。
它主要由 <b>Multi-Head Attention (MHA)</b>、<b>Feed Forward Network</b> 和 <b>LayerNorm</b> 组成。
对于输入序列的每一个元素 $x_i$，首先将其映射到 $d_q, d_k, d_v$ 维度，分别表示为 $q, k, v$：</p>

<center>$q_i, k_i, v_i = W_qx_i, W_kx_i, W_vx_i$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">然后使用 attention 关注自身与序列中的其他元素的关系来更新自身的表示：</p>

<center>\hat{x}_i = \sum_{i=1}^N{p(q,k_i)v_i}, i=[1,...,n]$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">任意看出，上述的操作可以将 $x$ 视为一个矩阵，使得各个元素同时计算计算，即：</p>

<center>$\hat{x} = Attention(Q,K,V) = softmax(QK^T)V = softmax((W_qx)(W_kx)^T)W_vx$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">由于 $QK^T$ 的数量级 $\approx d_q \times d_k \approx d_k^2$，为防止计算数值过大，本文选择将乘积结果 $\times \frac{1}{\sqrt{d_k}}$ 进行缩放，
因此，attention 的表达式为：</p>

<center>$\hat{x} = Attention(Q,K,V) = softmax(\dfrac{QK^T}{\sqrt{d_k}})V$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">在</p>
