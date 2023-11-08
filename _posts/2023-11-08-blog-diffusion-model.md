---
title: 'The Basic Knowledge of Diffusion Model (DM)'
date: 23-11-08
permalink: /posts/2023/11/blog-diffusion-model/
tags:
  - 深度学习基础知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客参考了<a href="https://zhuanlan.zhihu.com/p/663880249?utm_psn=1705611921401720833" target="_blank">
DDPM讲解</a>，详细讲述了最近大火的 DM 模型的数学原理/推导及编程。</p>

DM的基本原理
===

<p style="text-align:justify; text-justify:inter-ideograph;">DM 的思想如下：往一张图像中一点一点地加噪声，经过无限次之后，它将变成一个各向同性的标准正态分布噪声，这个过程叫做扩散过程。
那么将这个过程进行反转，往一个各向同性的标准正态分布噪声中一点一点地加上<b>特定的</b>噪声(即加上扩散过程每一步噪声的相反数)，那么经过无限次之后，
它就会变回到原始的图像，这个过程叫做逆扩散过程。</p>

![DDPM](/images/DDPM.png)

<p style="text-align:justify; text-justify:inter-ideograph;">具体而言，假设每一步的噪声 $d_t \in \mathcal{N}(0, \beta_t\boldsymbol{I)$，
则 $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1},\beta_t\boldsymbol{I}) \Rightarrow x_t = \sqrt{1 - \beta_t}x_{t-1} + \sqrt{\beta_t}\varepsilon_{t-1}, \varepsilon_{t-1} \in \mathcal{N}(0, 1), \beta_1 < ... < \beta_T \Rightarrow x_T \sim \mathcal{N}(0, \boldsymbol{I})$，
即 $x_{t}^2 \Rightarrow (\sqrt{1-\beta_t}x_{t-1})^2 + {d_t}^2$ \Rightarrow (\sqrt{1-\beta_t}x_{t-1})^2 + {\sqrt{\beta_t}\varepsilon_t}^2$。</p>