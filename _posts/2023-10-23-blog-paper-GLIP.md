---
title: 'GLIP'
date: 23-10-23
permalink: /posts/2023/10/blog-paper-glip/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://openaccess.thecvf.com/content/CVPR2022/html/Li_Grounded_Language-Image_Pre-Training_CVPR_2022_paper.html?ref=blog.roboflow.com" target="_blank" title="GLIP">Grounded Language-Image Pre-training</a></p>

发表会议：The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR 2022)

第一作者：Liunian Harold Li (University of California at Los Angeles (UCLA))

Question
===

<p style="text-align:justify; text-justify:inter-ideograph;">如何使得目标检测模型学习到更细粒度的，language-aware 的图像理解(即 object-level 的视觉表征)</p>

Method
===

![GLIP architecture](/images/paper_GLIP_architecture.png)

<p style="text-align:justify; text-justify:inter-ideograph;">主流的目标检测模型包括一阶段和两阶段的方法。例如，两阶段的方法首先使用 RPN 网络生成大量的 anchor (predicted box)，
然后对每个 anchor 学习分类器(box classifier)和定位器(box regressor)。
这就存在 3 个问题：1) 使用该架构的目标检测方法只能检测出事先给定的类别，而无法做到 open-vocabulary 的目标检测；
2) 目前而言，所有目标检测模型的类别数太少，一般不超过 2000 类，无法使得模型推广到更大的范围进行使用；
3) 这种仅仅将类别 index 作为最终分类(在训练时是将一个个类别标号为 0,1,...然后使用 Cross-entropy损失进行学习的，并没有使用/学习到语义信息)的学习方法，
不能很好地让模型学习到 language-aware 的图像表征。
为此，本文对目标检测模型进行了改进。首先是针对问题 1，借鉴于 CLIP 在分类上的 open-vocabulary 的思路(使用双塔结构计算相似度进行分类)，
GLIP 也使用了双塔结构并计算相似度进行目标检测。具体而言，如图，GLIP 分别使用了 Text Encoder $Enc_L$ 和 Image Encoder $Enc_I$ 对类别和图像进行编码。
在 Text-Encoder 部分，GLIP 通过一定的构造方式将类别转化为句子 $Prompt$ (例如，将每个类别按顺序进行排列并使用 . 进行分隔，就形成了句子)，
然后将 $Prompt$ 送入 Text-Encoder $Enc_L$ 进行编码得到特征 $P \in R^{M \times d}$，其中 $M$ 表示 $Prompt$ 中的单词数。
在 Image-Encoder 部分，GLIP 首先通过一定的方式获得大量的 box/region (例如可以使用预训练好的 RPN)，
然后将其全部送入 Image-Encoder $Enc_I$ 进行编码得到特征 $O \in R^{N \times d}$，其中 $N$ 表示 box 的个数。</p>

<ul><li><p style="text-align:justify; text-justify:inter-ideograph;">对于分类，GLIP 使用点乘计算它们之间的相似度：$S_{ground} = OP^T$ \in R^{N \times M}。
最后再使用 cross-entropy/focal loss 计算损失：$L_cls = loss(S_{ground}, T), T \in \{0,1\}^{N \times c}$，
其中 $c$ 表示类别数，$T$ 可以通过 多对一匹配或者二部匈牙利算法获得。
可以看到 $S_{ground} \in R^{N \times M}$ 与 $T \in \{0,1\}^{N \times c}$ 并不匹配，通常而言 $M > c$ (因为句子中包括了很多其他的词和符号，如 .，此外一个类别还可能存在多个词等)。
所以需要将 $T$ 进行扩展，对于 focal loss，将一个类别中的所有词都规定为 positive match，并把其他额外的词都视为 negative match；对于 cross-entropy loss，
将所有没被算法(多对一匹配或者二部匈牙利算法)匹配到类别的，即 no positive match，统一将它们匹配给 $[NoObj]$ token (这是加在句子末尾表示句子结束的词)，最终获得 $T'$ 送入到 $L_cls$ 计算损失。</p></li>

<li><p style="text-align:justify; text-justify:inter-ideograph;">而对于定位，GLIP和主流的目标检测方法一致，都是将每个 box 特征 $O_i$ 进行 ROIPool/ROIAlign，
然后送入下游网络(CNN+Linear layer)对 box 的位置偏差 $(\Delta x, \Delta y, \Delta w, \Delta h)$ 进行预测：$D_{predict} = Linear(CNN(ROIPool(O))) \in R^{N \times 4}$。
最后使用 $L_1/L_2$ loss 计算损失：$L_{loc} = \sum_{i=1}^{N}{||D_{predict}^i - D_{ground}^i||_2^2}$，
其中 $D_{ground} \in R^{N \times 4}$，表示每个 box 的真实偏差，对于没有匹配到类别的 box，其真实偏差全为 $0$。</p></li></ul>

