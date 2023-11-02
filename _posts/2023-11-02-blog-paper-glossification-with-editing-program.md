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

<p style="text-align:justify; text-justify:inter-ideograph;">可以看到，glossification 是一个 Machine Tranlation 问题。最直观的做法是使用 encoder-decoder 架构，输入 sentence，输出预测的 glosses。
但是，和普通的 Machine Translation 不同的是：一方面， sentence-glosses 的数据集较少，如果直接采用传统的 Machine Translation 方式可能效果不佳；另一方面，和传统的 Machine Translation 不同，
sentence 和 glosses 的字典集是相同的，仅仅是语法规则不同，这就意味着它们之间存在着很强的句法联系。因此，本文便通过利用它们之间的句法联系作为先验知识来帮助模型学习，以减轻数据集匮乏的问题。
通过观察可以看到，sentence 和 glosses 不仅字典集相同，而且在每个 sentence-glosses 对中，glosses 的大部分单词都在对应的 sentence 中出现过(通常是保留关键词，删除次要词)。
因此，相比于直接预测 glosses，可以通过对 sentence 进行增删改操作(即 editing actions)来获得对应的 glosses。
这样，通过显式地引入转化过程可以更好地帮助模型学习(即之前需要模型自己摸索如何从 sentence 转化到 glosses，现在通过一步步的 editing action 显式地告诉模型转化规则)。
所以，模型不再直接预测 glosses，而是预测 editing program，并执行它以获得最终的 glosses。
具体而言，本文设计了特定的 editing program 语法，如下图所示：</p>

![syntax](/images/paper_glossification_editing_program_syntax.png)

