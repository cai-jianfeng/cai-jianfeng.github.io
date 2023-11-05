---
title: 'SLTUnet'
date: 23-11-05
permalink: /posts/2023/11/blog-paper-sltunet/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://openreview.net/forum?id=EBS4C77p_5S" target="_blank" title="SLTUnet">SLTUNET: A Simple Unified Model for Sign Language Translation</a></p>

<p style="text-align:justify; text-justify:inter-ideograph;">发表会议：International Conference on Learning Representations (ICLR 2023)</p>

第一作者：Biao Zhang (University of Edinburgh)

Question
===

<p style="text-align:justify; text-justify:inter-ideograph;">如何解决 Sign Language Translation (SLT) 的数据缺乏和模态差异(video & text)问题</p>

Preliminary
===
<p style="text-align:justify; text-justify:inter-ideograph;">SLT：SLT 是要将手语视频(sign language video)翻译为对应意思的口语语句(spoken language text)。
通常有 $2$ 中实现方法：</p>

<p style="text-align:justify; text-justify:inter-ideograph;">1) $cascading$：它依赖于其他的中间表示，如 $glosses$ 
(gloss 是手语的一种书面形式，它通常是通过手语视频翻译而来，其中每个词语和手语视频中的每个动作都一一对应。
通常而言，gloss 会比 sentence (即我们日常说话使用的语句)更加简洁，同时因为它是按手语视频的动作及顺序翻译而来的，因此和 sentence 拥有不同的语法规则，
但是它拥有和 sentence 相似甚至相同的字典集(即它们所用的词汇表是相似/相同的)。)。
然后将 SLT 任务分解成 $2$ 个子任务：$sign\ language\ recognition$ 和 $gloss-to-text\ translation$。
其中，$sign\ language\ recognition$ 是将手语视频翻译为 $glosses$ 序列(Sign2Gloss)；而 $gloss-to-text\ translation$ 则是将 $glosses$ 序列转化为对应的口语语句。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">2) $end-to-end$：它直接单个模型将手语视频翻译为对应的口语语句，而不需要多阶段翻译。</p>

Method
===
<p style="text-align:justify; text-justify:inter-ideograph;"></p>