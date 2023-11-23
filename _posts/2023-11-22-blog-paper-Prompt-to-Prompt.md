---
title: 'Prompt-to-Prompt'
date: 23-11-22
permalink: /posts/2023/11/blog-paper-prompt-to-prompt/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://openreview.net/forum?id=_CDixzkzeyb" target="_blank" title="Prompt-to-Prompt">Prompt-to-Prompt Image Editing with Cross-Attention Control</a></p>

<p style="text-align:justify; text-justify:inter-ideograph;">发表会议：International Conference on Learning Representations (ICLR 2023)</p>

<p style="text-align:justify; text-justify:inter-ideograph;">第一作者：Amir Hertz (Google Research)</p>

Question
===

<p style="text-align:justify; text-justify:inter-ideograph;">如何提高 Instruction-based image editing model 的性能(即仅仅使用 Instruction/text 来对输入的图像进行编辑，
例如：输入文字“将图像的 xx 物体去掉/在 xx 位置添加一个 xx 物体”和一张给定的图像，就可以获得按照文字要求编辑后的图像)</p>

Method
===

<p style="text-align:justify; text-justify:inter-ideograph;">针对图像编辑最常见的方法是使用 mask 来显式确定需要编辑的区域，然后使用不同的方法对 mask 内的区域进行编辑。
但是 mask 过程很繁琐，不如直接文本驱动的编辑。此外，对图像内容进行 mask 会消除重要的结构信息，使得在后面修复过程中完全看不到先前的信息。
因此，本文提出一种仅使用文本进行编辑的方法，通过 <b>Prompt-to-Prompt</b> 的操作，在预训练的 text-to-image 扩散模型中对图像进行语义编辑。
通过深入研究交叉注意力层(cross-attention)，并探索其控制生成图像的语义强度(即通过定向改变 cross-attention 能多大程度上控制图像的变化)。
本文主要考虑 U-net 内部的 cross-attention maps，这是一个高维张量，结合了从提示文本中提取的 token 和需要生成的图像的 pixel 之间的对应关系。
这些 maps 包含丰富的语义关系，对生成的图像有重要影响。
因此，本文提出一个简单直观的想法：可以通过在扩散过程中注入不同的交叉注意力图来编辑图像，以在指定的扩散步骤中，控制每个 pixel 关注提示文本的对应 token。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">不同与传统的方法(给定一张原始图像和编辑指令)，本文需要给定原始图像(即需要编辑的图像)的标题 $\mathcal{P}$，以及目标图像(即编辑完成的图像)的标题 $\mathcal{P}^*$。
这就需要将编辑操作写入到 $\mathcal{P}$ 以获得 $\mathcal{P}^*$。本文关注的编辑操作(常见的)包括 $3$ 种：<b>change</b> (替换)，即将  $\mathcal{P}$ 中的某个词替换为另一个词；
<b>Add</b> (添加)，即在  $\mathcal{P}$ 中添加新词(对于删除词只需要将 $\mathcal{P}$ 变成目标图像标题，$\mathcal{P}^*$ 变成原始图像标题就可以变成添加词)；
<b>Scale</b> (缩放)，即放大/缩小 $\mathcal{P}$ 中的某个词的表现程度。
假设 $\mathcal{I}$ 表示使用现成的 text-to-image DM 模型输入 $\mathcal{P}$ (和随机种子 $s$)生成的图像。
本文的目标是仅使用编辑提示 $\mathcal{P}^*$ 的指导，编辑图像 $\mathcal{I}$，以获得编辑后的图像 $\mathcal{I}^*$，它保持了原始图像的内容和结构，仅修改对应的编辑提示要求的内容。
一种简单的方法是固定随机种子($=s$)，然后使用同样的 text-to-image DM 模型输入 $\mathcal{P}^*$ 再次生成图像作为编辑后的图像，但是这样会导致整张图像的全部变动。
这说明仅仅固定随机种子还是无法解决模型生成的随机性。为此，本文便通过固定 U-net 内部的 cross-attention maps 来进一步限制模型的输出。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">具体而言，</p>