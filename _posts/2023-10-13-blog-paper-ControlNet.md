---
title: 'ControlNet'
date: 23-10-13
permalink: /posts/2023/10/blog-paper-controlnet/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html" target="_blank" title="ControlNet">Adding Conditional Control to Text-to-Image Diffusion Models</a></p>

发表会议：International Conference on Computer Vision(ICCV 2023, Best Paper(Marr Prize))

第一作者：Lvmin Zhang(PhD with Stanford University)

Question
===
<p style="text-align:justify; text-justify:inter-ideograph;"> 如何在 Text-to-Image Diffusion 图像生成模型中添加条件控制其图像的生成(这里的条件主要包括 visual condition，如 mask, depth等，即输入的条件也是图像)，同时保证其生成图像的逼真性。 </p>

Preliminary
===
<p style="text-align:justify; text-justify:inter-ideograph;"> Diffusion Model：扩散模型 </p>

Method
===
<p style="text-align:justify; text-justify:inter-ideograph;"> 这个问题最简单的方法是找到一个带有该条件(假设为 $C_i$)的数据集，再在别人已经预训练好的 Difussion 模型上进行微调。
但是这么做有一个问题：因为预训练的 Diffusion Model 是在大量图像上训练而来，如果直接将整个模型直接进行有监督微调，可能会使得模型生成效果大打折扣(即文中说的 overfitting 和 catastrophic forgetting)。
所以本文解决的是在加入了条件 $C_i$ 的情况下，既能输出符合条件 $C_i$ 的图像，又能保证其逼真度；同时，本文还解决了多个 $C_i$ 同时作用于一张图像的生成问题，并且训练的参数也较少。</p>