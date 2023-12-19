---
title: 'The Basic Knowledge of Automatic Mixed Precision'
date: 23-12-19
permalink: /posts/2023/12/blog-torch-amp/
star: superior
tags:
  - 深度学习基本知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要讲解了使用自动混合精度(AMP)降低模型内存占用的原理和具体实现。</p>

Automatic Mixed Precision
===

<p style="text-align:justify; text-justify:inter-ideograph;">通常而言，<b>AMP</b> 主要包括 $2$ 个部分：<b>autocast</b> 和 <b>grad scale</b>。</p>