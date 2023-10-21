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

<p style="text-align:justify; text-justify:inter-ideograph;">本文就针对这 2 个 问题分别给出了自己的解决思路。</p>