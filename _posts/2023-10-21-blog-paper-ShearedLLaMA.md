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

