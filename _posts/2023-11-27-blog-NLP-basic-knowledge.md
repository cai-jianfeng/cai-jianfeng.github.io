---
title: 'The Basic Knowledge of NLP'
data: 23-11-27
permalink: '/posts/2023/11/blog-NLP-basic-knowledge'
tags:
  - 深度学习基础知识
---

<h1>Metric</h1>

<p style="text-align:justify; text-justify:inter-ideograph;"><b>BLEU</b>：Bilingual Evaluation Understudy，主要用于计算 Machine Translation 的质量。
计算公式如下：</p>

$$BLEU_{n-gram} = BP \times exp(P_n),\ BLEU = BP \times exp(\dfrac{\sum_{i=1}^NP_n}{N}), \\ 
P_n = \dfrac{\sum_{n-gram \in \hat{y}}{Counter_{Clip}(n-gram)}}{\sum_{n-gram \in \hat{y}}{Counter(n-gram)}}, 
BP = \begin{cases}1, & L_{out} > L_{ref} \\ exp(1 - \dfrac{L_{ref}}{L_{out}}), & otherwise \end{cases}$$
