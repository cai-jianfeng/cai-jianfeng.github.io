---
title: 'XmDA'
date: 23-11-05
permalink: /posts/2023/11/blog-paper-xmda/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://arxiv.org/abs/2305.11096" target="_blank" title="XmDA">Cross-modality Data Augmentation for End-to-End Sign Language Translation</a></p>

<p style="text-align:justify; text-justify:inter-ideograph;">发表会议：截至2023年11月6日暂无，论文版本为 arxiv-v3</p>

第一作者：Jinhui Ye (HKUST(GZ))

Question
===

<p style="text-align:justify; text-justify:inter-ideograph;">如何解决 Sign Language Translation (SLGT) 的数据缺乏和模态差异(video & text)问题</p>

Preliminary
===

<p style="text-align:justify; text-justify:inter-ideograph;">SLT：SLT 是要将手语视频(sign language video)翻译为对应意思的口语语句(spoken language text)。详见 <a href="https://cai-jianfeng.github.io/posts/2023/11/blog-paper-sltunet/" target="_blank">SLTUnet</a> 的 Preliminary。</p>

Method
===

<p style="text-align:justify; text-justify:inter-ideograph;">与 <a href="https://cai-jianfeng.github.io/posts/2023/11/blog-paper-sltunet/" target="_blank">SLTUnet</a> 相似，
本文也想通过合理利用 $glosses$ 数据来帮助 end-to-end 模型学习以减轻数据缺乏问题，同时使得模型能更好地学习模态融合特征。
因此，本文提出了 <b>Cross-modality Mix-up</b> 模块 和 <b>Cross-modality Knowledge Distillation</b> 模块来帮助模型学习。
假设 SLT 数据集为 $\mathcal{D} = \{(S_i, G_i, T_i)\}_{i=1}^N$，其中 $S_i = \{s_z\}_{z=1}^Z$ 表示手语视频，$G_i = \{g_v\}_{v=1}^V$ 表示对应的 $glosses$，
$T_i = \{t_u\}_{u=1}^U$ 表示对应的 sentence。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">本文使用的模型为 Sign Language Transformers。它主要包括 $5$ 个部分。
<b>sign embedding</b> 将手语视频编码为 embedding，本文使用预训练的 SMKD 模型，然后将编码后的 embedding 通过一层线性层投射到特定维度(与 gloss embedding 相同)。
接着，<b>Translation Encoder</b> 将 sign embedding 编码为 contextual representations $h(S)$。
然后，将 $h(S)$ 输入到 <b>Translation Decoder</b> 输出预测的 sentence $\hat{T}$，并使用 cross-entropy 损失进行训练：</p>

<center>$L_{MLE} = -\sum_{u=1}^{|T|}{log\ \mathcal{P}(t_u|t_{<u},h(S))}$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">此外，Sign Language Transformers 还包括 <b>Gloss Embedding</b> 和 <b>CTC Classifer</b> 模块。
其中， Gloss Embedding 类似于 word embedding，它使用一个 embedding 矩阵将 $gloss$ token 转化为向量表示。
而 CTC Classifier 是将 $gloss$ 作为中间的监督信号来训练 Translation Encoder。具体而言，它在 Translation Encoder 上增加了一个线性层和 softmax 函数来预测手语视频每一帧的 gloss 概率分布 $\mathcal{P}(g_z|h(S)), z\in [1,...,Z]$。
然后通过边缘化 $S$ 到 $G$ 之间的所有有效映射来建模概率分布：</p>

<center>$\mathcal{P}(G^*|h(S)) = \sum_{\pi \in \Pi}{\mathcal{P}(\pi|h(S))}$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">其中，$G^*$ 是 ground-truth，$\pi$ 是预测的 $glosses$，而 $\Pi$ 是所有合法的 $glosses$ 集合。最后，CTC 损失函数为 $L_{CTC} = 1 - \mathcal{P}(G^*|h(S))$。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">对于 Cross-modality Mix-up 模块，它的主要作用是对齐 sign embedding 和 gloss embedding。
具体而言，Sign Language Transformers 的 Sign Embedding 将 $S = \{s_z\}_{z=1}^Z$ 编码为 sign embedding $\mathcal{F} = [f_1,...,f_Z]$，
而 Gloss Embedding 将 $G = \{g_v\}_{v=1}^V$ 映射为 gloss embedding $\mathcal{E} = [e_1,...,e_V]$。
Cross-modality Mix-up 通过将 $\mathcal{F}$ 和 $\mahcal{E}$ 结合在一起获得混合模态的 embedding $\mathcal{M}$。
首先，本文使用 CTC classifier 作为 sign-gloss forced aligner，通过最大化 $\pi$ 的边缘概率来计算每个 gloss token $g_v$ 对应的手语视频的起始点 $l_v$ 和终止点 $r_v$：</p>

<center>$\pi^* \Leftarrow \overset{argm}{\pi \in \Pi} max \mathcal{P}(\pi|h(S)) \Leftarrow = \overset{argm}{\pi \in \Pi} max \sum_{v=0}^{V}\sum_{z = l_v}^{r_v} log\mathcal{P}(g_z = g_v^*)$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">在求解得每个 $g_v$ 对应的 $l_v$ 和 $r_v$ 之后，通过一个预定义的阈值 $\lambda$ 来混合 $\mathcal{F}$ 和 $\mathcal{e}$ 以获得 $\mathcal{M}$：</p>

<center>$m_v = \begin{cases}\mathcal{F}[l_v:r_v],\ p \leq \lambda \\ \mathcal{E},\ p > \lambda \end{cases}, p \in \mathcal{N}(0,1);\ \mathcal{M} = [m_1,..m_V]$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">注意，虽然 $\mathcal{M}$ 的下标标记到 $V$，但是其实际上的元素个数为 $Z$，因为其中的一部分 $m_i$ 的元素个数不是 $1$，而是 $r_v - l_v + 1$ 个。</p>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>


