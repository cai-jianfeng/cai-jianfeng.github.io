---
title: 'Glossification with Editing Program'
date: 23-11-02
permalink: /posts/2023/11/blog-paper-glossification-with-editingprogram/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://ojs.aaai.org/index.php/AAAI/article/view/21457" target="_blank" title="Glossification with Editing Program">Transcribing Natural Languages for the Deaf via Neural Editing Programs</a></p>

<p style="text-align:justify; text-justify:inter-ideograph;">发表会议：Association for the Advancement of Artificial Intelligence (AAAI 2022)</p>

第一作者：Dongxu Li (Australian National University, currently in Salesforce AI)

Question
===

<p style="text-align:justify; text-justify:inter-ideograph;">如何通过有效挖掘 sentence 和 glossed 之间的句法联系来提高 glossification 的准确率。</p>

Preliminary
===

<p style="text-align:justify; text-justify:inter-ideograph;">glossification：gloss 是手语的一种书面形式，它通常是通过手语视频翻译而来，其中每个词语和手语视频中的每个动作都一一对应。
通常而言，gloss 会比 sentence (即我们日常说话使用的语句)更加简洁，同时因为它是按手语视频的动作及顺序翻译而来的，因此和 sentence 拥有不同的语法规则，
但是它拥有和 sentence 相似甚至相同的字典集(即它们所用的词汇表是相似/相同的)。
例如，对于正常的 sentence "Do you like to watch baseball games?"，表达成 gloss 就变成了 "baseball watch you like?"。
而 glossification 任务就是要将正常的 sentence 翻译为 gloss，以便后续继续翻译为手语视频。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">Editing Program：对于一个给定序列 $seq_o$，可以通过一系列 editing actions 操作将其变成另一个序列 $seq_t$。
其中，editing actions 包括 1) $Add\ e\ i$，即在序列 $seq_o$ 的第 $i$ 个位置添加元素 $e$；
2) $Del i$，即删除序列 $seq_o$ 的第 $i$ 个位置的元素；
3) $Sub\ e\ i$，即将序列 $seq_o$ 的第 $i$ 个位置的元素替换成元素 $e$。
而将序列 $seq_o$ 变成另一个序列 $seq_t$ 的一系列 editing actions 操作称为 Eiditing Program。</p>

Method
===

<p style="text-align:justify; text-justify:inter-ideograph;"></p>