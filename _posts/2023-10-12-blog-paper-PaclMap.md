---
title: 'PaclMap'
date: 23-10-12
permalink: /posts/2023/09/blog-paper-paclmap/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://ieeexplore.ieee.org/abstract/document/9944792" target="_blank" title="PaclMap">Multipattern Mining Using Pattern-Level Contrastive Learning and Multipattern Activation Map</a></p>

发表期刊：IEEE Transactions on Neural Networks and Learning Systems(TNNLS, 2023年 CCF B)

第一作者：Xuefeng Liang(Professor with Xidian University)；h指数 18(2023)

Question
===
文章主要解决的问题是如何在不知道 visual pattern 的 label 和 number 的情况下找出某一类图像的所有 visual patterns。

Preliminary
===
<p style="text-align:justify; text-justify:inter-ideograph;"> visual pattern：它是一种临界于 pixel 和 category label 之间的语义特征，即它的语义信息比单纯的像素高，但是又低于图像本身的类别。一类图像中可以包含多个 visual patterns，它们都独特地描述了该类图像(即其他图像不存在或者不是其他图像的显著性特征)。
一个通俗直观的例子是，对于西安的鼓楼，所有的 鼓楼 照片都有 category label —— 鼓楼；而鼓楼中含有其标志性代表 鼓 和 楼，它们两个就是 鼓楼 这类图像的 visual pattern。
虽然其不一定都在每张 鼓楼 照片中出现，但是每张照片都至少出现一个 visual patter，且统计下来，这两个 visual pattern 在 鼓楼 照片中出现的次数远超其他类别,同时它们又都带有一定的语义信息，所以它们便是 鼓楼 类照片的 visual patterns。
然而对于更加广泛的 visual pattern 来说，它们的语义信息可能不如前面那个例子那么直观，例如纹理等，一方面给每类图像标记 visual pattern 标签是一类难题；另一方面人工寻找每类图像的每个 visual pattern 也是一类难题。
因此，现在的方法大部分都使用模型输出的 feature 作为每类图像的 visual pattern(例如 CNN 卷积层输出的 activation)。
虽然其相较 鼓、楼 这种实质性标签难以理解，但是好处是不需要确定其具体含义(这样就不需要费力进行 visual pattern 标签标记)，同时其也含有和实质性标签本质一样的语义信息。</p>

<p style="text-align:justify; text-justify:inter-ideograph;"> Contrastive Learning：对比学习是最近十分火热的无监督学习方法，它通过构造 self-supervision 来实现网络训练。
大体上，对比学习采用两个网络(分别叫 online network 和 target network)进行训练，online network 输入原始图像，其输出一段 vector 作为特征 $f_1$，
而 target network 输入对应原始图像的正样本(它与原始图像互为正样本对，通常是原始图像通过数据增强得来，比如旋转)，其也输出一段 vector 作为 $f_1$ 的正样本特征 $f_2$。
对于 $f_1$ 和 $f_2$，我们希望它们两的特征应该相似，所以可以使得它们的相似度函数 $sim(f_1, f_2)$(最简单的是点乘，然后正则化：$\frac{f_1f_2^T}{\sqrt{f_1f_1^T}\sqrt{f_2f_2^T}}$)的值接近 1。
然而这种方法有一定的弊端， 那就是模型可能会找到一条”捷径“，online network 和 target network 对于每一个输入，都输出相同的 vector(比如全 0)，这样就可以保证每个正样本对之间的相似度为 1。
因此，我们需要引入负样本(通常是和原始图像不同的任意其他图像，其通过 target network 的输出向量为 $f_3$)来限制模型的输出。
我们希望 $f_1$ 和 $f_3$ 应该不相似，所以可以使得它们的相似度函数 $sim(f_1, f_3)$ 的值接近 0。
通常而言每个原始图像的正样本就一个，而负样本可以很多个(一个直观的理解是对于负样本，模型可以很容易判断其和原始图像不相似，如果就只使用一个负样本，则会导致负样本基本上没贡献)。
这就和传统的分类问题较为相似(把正确的类别看作正样本，其他类别均看作负样本)，
因此可以使用 BCE loss 进行训练：$-log\frac{exp(sim(f_i,f_+)/\tau)}{\sum_{j,f_j \neq f_+}^{N}{exp(sim(f_i, f_j)\tau)}+exp(sim(f_i,f_+)/\tau}$。
其中 $\tau$ 表示温度因子，其越大则使得不相似的负样本的影响越小(可以简单证明：假设 $sim(f_o, f_+) = 1$, $sim(f_o, f_{-1}) = 0.9$， $sim(f_o, f_{-2}) = 0.1$；

最终训练出来的 online network 具有很好的模型先验，可以送到下游任务进行微调使用。</p>

Method
===
<p style="text-align:justify; text-justify:inter-ideograph;"> 在之前的方法中，它们都假设每类图像只存在一个 visual pattern，
这样就可以通过构造 category label 到 visual pattern 之间的一一映射以获得 visual pattern。
但是本文认为每类图像不止一个 visual pattern，需要在仅仅知道 category label 的情况下将它们都找出来。
一般来说，visual pattern包含两个特点：discrimination 和 frequency。
discrimination 表示其具有可判别性，而不是笼统的语义信息(例如 山 这个语义信息就具有笼统性)；frequency 表示其应该在该类图像中频繁出现。
因此，针对 discrimination，本文提出了 Pacl 模块