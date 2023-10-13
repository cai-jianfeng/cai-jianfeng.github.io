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
<p style="text-align:justify; text-justify:inter-ideograph;"> 如何在 Text-to-Image Diffusion 图像生成模型中添加条件控制其图像的生成(这里的条件主要包括 visual condition，如 mask, depth等，即输入的条件也是图像) </p>

Preliminary
===
<p style="text-align:justify; text-justify:inter-ideograph;"> Diffusion Model：扩散模型 </p>