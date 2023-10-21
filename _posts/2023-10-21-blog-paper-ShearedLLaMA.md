---
title: 'ShearedLLaMA'
date: 23-10-21
permalink: /posts/2023/10/blog-paper-shearedllama/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://arxiv.org/abs/2310.06694" target="_blank" title="ShearedLLaMA">Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning</a></p>

发表会议：截至2023年10月21日暂无，论文版本为 arxiv-v1

第一作者：Mengzhou Xia(Princeton University)

Question
===

<p style="text-align:justify; text-justify:inter-ideograph;">如何通过利用现有的 pre-trained LLM(大模型，如 LLaMA2-7B)，得到一个较小的(如 1.3B)且性能和同等规模的模型(如 LLaMA2-1.3B)相似的 LLM, 
并且使用的计算量比从头训练一个同等规模的 LLM 少得多</p>

Method
===

![targeted structured pruning](/images/paper_ShearedLLaMA.png)

<p style="text-align:justify; text-justify:inter-ideograph;">首先要从一个以及训练好的大模型获得高性能的小模型最直观也是最主流的方法就是模型剪枝(pruning；当然 student-teacher 架构也是一种解决方法，但是需要训练 student 模型 from scratch)。
但是 pruning 有 2 个问题需要解决：</p>

1) 如何设计小模型的架构，使其拥有良好的性能和高效推理的潜力？ 
2) 如何继续训练剪枝好的小模型使其达到理想性能？

<p style="text-align:justify; text-justify:inter-ideograph;">本文就针对这 2 个 问题分别给出了自己的解决思路。
首先是 模型的架构，本文观察到现有的 structed pruning 方法往往会导致模型架构偏离流行的架构，并且作者假设现有的模型架构一般都是经过重重验证的，在各方面的性能潜力应该都是不错的。
与其使用无约束的 structed pruning 并承担导致模型架构畸形的风险，不如添加约束使得 structed pruning 使得模型剪枝到指定的模型架构上。
具体而言，本文提出了 target structed pruning，通过在不同粒度上学习大模型参数上的一组剪枝掩码(pruning mask)——从 global，如 layers 和 hidden dimentions，到 local，如 attention heads 和 intermediate dimensions。
以普通的 Transformer为例，其中，layers 就是 对每一个 transformer block 都学习一个掩码数值 $0$ 或 $1$ 来选择是否保留该 block；
hidden dimentions 就是 Embedding 层将原始数据转换成的潜表示所包含的维度数量(即通道数)，通过给每个维度学习一个掩码数值来选择是否保留该维度，
之所以它们俩是 global 的，是因为它们是在整个模型角度来进行考虑的。
而 attention heads 就是对每个 Transformer block 中的每一个 attention head 都学习一个掩码数值来选择是否保留；
而 intermediate dimensions 就是每个 Transformer block 中的 FFN 的中间表示的维度数量，通过给每个 Transformer block 中的 FFN 的中间表示的每个维度学习一个掩码数值来选择是否保留该维度，
之所以它们俩是 local 的，是因为它们是在每个 Transformer block 角度来进行考虑的。
假设源模型，即大模型 $M_S$ 的 hidden dimention 为 $d_S$, 有 $L_S$ 层，每一层包括 $1$ 个 Multi-head Attention (MHA) 和一个 FFN，
每个 MHA 中 head 的数量为 $H_S$，每个 FFN 的 intermediate dimensions 为 $m_S$，则可以设计如下掩码：</p>

|  Granularity  | Layer | Hidden dimension | Head |       Intermediate dimension       |
|:-------------:|:-------:|:-------:|:-------:|:----------------------------------:|
| Pruning masks |$z^{layer} \in R^{L_S}$|$z^{hidden} \in R^{d_S}$|$z^{head} \in R^{H_S} (\times L_S)$| $z^{int} \in R^{m_S} (\times L_S)$ |

<p style="text-align:justify; text-justify:inter-ideograph;">例如，其中的 $z^{layer}$ 表示 $L_S$ 层 Transforner Block 的掩码，$z_i^{layer} = 0$ 表示 $i$ 层的 Transforner Block 舍弃；反之则表示保留。
接着便是如何将掩码融入到剪枝优化过程中以限制模型的架构。很明显这是一个约束的优化问题，最常用的方法便是 Lagrange multipliers 法。
例如，假设目标模型，即小模型的每个 MHA 的 head 数量为 $H_T$，则关于以 $H_T$ 为约束条件的损失函数为：
$\widetilde{L}^{head}(\lambda, \theta, z) = \lambda^{head} · (\sum{z^{head}} - H_T) + \phi^{head} · (\sum{z^{head}} - H_T)^2$。
最终，通过联合优化模型参数(model weights；以模型性能为损失函数)和剪枝掩码(pruning masks)便可进行剪枝训练：</p>

<center>$min_{\theta,z}max_{\lambda,\phi}L_{prune}(\theta, z, \lambda, \phi)$</center>
<center>$L_{prune}(\theta, z, \lambda, \phi) = L(\theta, z) + \sum_{j=1}^{L_S}\widetilde{L}^{head} + \sum_{j=1}^{L_S}\widetilde{L}^{int} + \widetilde{L}^{layer} + \widetilde{L}^{hidden}$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">其中 $L(\theta, z)$ 表示 language modeling loss，使用的模型权重是经 pruning mask 掩码后的
(注意，这里的模型权重还都是从大模型直接复制而来的，在 target structed pruning 训练时更新的参数只是 pruning mask)。
由于每个 mask $z$ 都是 $[0,1]$ 之间的实数向量(和分数相似)，所以最终的选择策略是选择前 $H_T/L_T,d_T,m_T$ 高分数所对应的位置保留，其余丢弃。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">接着便是继续训练剪枝好的小模型，这里可能有一个疑问，为什么不直接在 target structed pruning 阶段让模型参数也一起训练更新？
首先，pruning 是一个相对较慢的过程(比标准的 LLM 训练慢 5 倍)，而本文发现 mask 的学习过程通过是很快收敛的，即很快就学习到了最终的 mask 模式。
其次，如果仅仅在 mask 收敛期间对模型参数进行更新，可能会适得其反，恶化模型的性能，所以便把模型参数的进一步学习单独进行。
训练剪枝模型的最直观方法是将训练大模型的数据集直接给小模型继续训练，直到达到预期性能。
但是这样有一个问题：对于大模型来说，它的训练是从头开始的，即它的初始状态是对各个 domain 的知识容量 $=0$；而对于剪枝得到的小模型，它本身对不同的 domain 存在一定的、各不相同的知识
(本文通过分析和实验证明了剪枝小模型一般对 low-entropy 且小型的 domain (例如 Github) 保留有较多的知识，而对于 high-entropy 且大型的 domain (例如 C4) 保留的知识较少)，
如果再直接将大模型训练的数据集给小模型训练，则会导致数据利用的 inefficient (即对于 low-entropy 且小型的 domain 学习的很快，但是对于 high-entropy 且大型的 domain 学的很慢)，
进而有损模型性能，所以就需要对数据集进行采样以平衡小模型在训练时对各个 domain 学习的快慢。
这里本文还是假设大模型 pre-trained 的最终模型结果是较为优质的
</p>