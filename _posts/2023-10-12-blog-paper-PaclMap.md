---
title: 'PaclMap'
date: 23-10-12
permalink: /posts/2023/09/blog-paper-paclmap/
tags:
  - 论文阅读
---

论文题目：[Multipattern Mining Using Pattern-Level Contrastive Learning and Multipattern Activation Map](https://ieeexplore.ieee.org/abstract/document/9944792)

发表期刊：IEEE Transactions on Neural Networks and Learning Systems(TNNLS, 2023年 CCF B)

第一作者：Xuefeng Liang(Professor with Xidian University)；h指数 18(2023)

Question
===
文章主要解决的问题是如何在不知道 visual pattern 的 label 和 number 的情况下找出某一类图像的所有 visual patterns。

Preliminary
===
<p style="text-align:justify; text-justify:inter-ideograph;">visual pattern：它是一种临界于 pixel 和 category label 之间的语义特征，即它的语义信息比单纯的像素高，但是又低于图像本身的类别。一类图像中可以包含多个 visual patterns，它们都独特地描述了该类图像(即其他图像不存在或者不是其他图像的显著性特征)。
一个通俗直观的例子是，对于西安的鼓楼，所有的 鼓楼 照片都有 category label —— 鼓楼；而鼓楼中含有其标志性代表 鼓 和 楼，它们两个就是 鼓楼 这类图像的 visual pattern。
虽然其不一定都在每张 鼓楼 照片中出现，但是每张照片都至少出现一个 visual patter，且统计下来，这两个 visual pattern 在 鼓楼 照片中出现的次数远超其他类别,同时它们又都带有一定的语义信息，所以它们便是 鼓楼 类照片的 visual patterns。
然而对于更加广泛的 visual pattern 来说，它们的语义信息可能不如前面那个例子那么直观，例如纹理等，一方面给每类图像标记 visual pattern 标签是一类难题；另一方面人工寻找每类图像的每个 visual pattern 也是一类难题。
因此，现在的方法大部分都使用模型输出的 feature 作为每类图像的 visual pattern(例如 CNN 卷积层输出的 activation)。
虽然其相较 鼓、楼 这种实质性标签难以理解，但是好处是不需要确定其具体含义(这样就不需要费力进行 visual pattern 标签标记)，同时其也含有和实质性标签本质一样的语义信息。</p>
