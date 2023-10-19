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