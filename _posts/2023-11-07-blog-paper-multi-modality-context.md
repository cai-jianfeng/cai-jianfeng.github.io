---
title: 'Multi-modality with Context'
date: 23-11-07
permalink: /posts/2023/11/blog-paper-multi-modality-with-context/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://openaccess.thecvf.com/content/ICCV2023W/ACVR/html/Sincan_Is_Context_all_you_Need_Scaling_Neural_Sign_Language_Translation_ICCVW_2023_paper.html" target="_blank" title="Multi-modality with Context">Is Context all you Need? Scaling Neural Sign Language Translation to Large Domains of Discourse</a></p>

<p style="text-align:justify; text-justify:inter-ideograph;">发表会议：International Conference on Computer Vision (ICCV 2023)</p>

第一作者：Ozge Mercanoglu Sincan (University of Surrey)

Question
===

<p style="text-align:justify; text-justify:inter-ideograph;">如何解决 Sign Language Translation (SLT) 的视觉歧义问题，即同一个手语动作在不同语境下的语义不同，而不同的手语动作也可能在不同的语境下表示相同的语义</p>


Method
===

![Multi-modality with Context architecture](/images/paper_multi_modal.png)

<p style="text-align:justify; text-justify:inter-ideograph;">本文借鉴了人类处理歧义词的方式，主要是依靠上下文的综合理解来确定当前词的语义。
为此，本文提出了将之前已经翻译好的 sentence 作为 context 来帮助当前的 sign language video 的翻译。
但是 PHOENIX14-T 和 CLS-Daily 数据集中包含的是一个个打乱顺序的 video-gloss-sentence 数据对，导致其失去了连贯性，无法找到每一对数据的前一数据对。
因此，本文抛弃了上述的 $2$ 个数据集，使用最近新发布的 BOBSL 和 DSGS 数据集，
它们包含的是一整个连贯的视频段($V_1,...,V_M$)以及所对应的翻译 sentence(S_1,...,S_N)，来利用它们的 context 进行学习。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">具体而言，给定一个手语视频 $V = (x_1,...,x_T)$，一个来自之前的上下文信息 $S_C = (S_{n-1},S_{n-2},...)$ 和一个 sparse sign spottings $S_p = (g_1,...,g_K)$ (即 $gloss$)。
模型需要输入 $V$，并在输入 $S_C$ 和 $S_p$ 进行辅助学习下，预测输出翻译的 sentence $S = (\omega_1,...,\omega_U)$。