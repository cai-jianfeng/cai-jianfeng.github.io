---
title: 'The Basic Knowledge of NLP'
data: 23-11-27
permalink: '/posts/2023/11/blog-NLP-basic-knowledge'
tags:
  - 深度学习基础知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要介绍了 NLP 任务中的基础知识，包括性能评价指标等。</p>

<h1>Metric</h1>

<p style="text-align:justify; text-justify:inter-ideograph;"><b><a href="https://aclanthology.org/P02-1040.pdf" target="_blank">BLEU</a></b>：Bilingual Evaluation Understudy，主要用于计算 Machine Translation 的质量。
计算公式如下：</p>

$$BLEU_{n-gram} = BP \times exp(log\ P_n),\ BLEU = BP \times exp(\dfrac{\sum_{i=1}^N\omega_nlog\ P_n}{N}), \\ 
P_n = \dfrac{\sum_{n-gram \in \hat{y}}{Counter_{Clip}(n-gram)}}{\sum_{n-gram \in \hat{y}}{Counter(n-gram)}}, 
BP = \begin{cases}1, & L_{out} > L_{ref} \\ exp(1 - \dfrac{L_{ref}}{L_{out}}), & L_{out} \leq L_{ref} \end{cases}$$

<p style="text-align:justify; text-justify:inter-ideograph;">其中，$Out$ 表示预测的句子 $\hat{y}$，$Ref$ 表示 ground-truth 的句子 $y$；$n-gram$ 表示由 $n$ 个词组成的词组；
$Counter_{Clip}(x)$ 表示 $x$ 在 $Ref$ 中出现的次数和在 $Out$ 中出现的次数的最小值；
$Counter(x)$ 表示 $x$ 在 $Out$ 中出现的次数；$L_{out/ref}$ 表示 $Out/Ref$ 的长度。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">对于整个测试集，BLEU 的计算公式如下：</p>

$$p_n = \dfrac{\underset{\mathcal{C} \in \{Candidates\}}{\sum}\underset{n-gram \in \mathcal{C}}{\sum} Count_{clip}(n-gram)}{\underset{\mathcal{C‘} \in \{Candidates\}}{\sum}\underset{n-gram‘ \in \mathcal{C’}}{\sum} Count(n-gram’)}, BP = \begin{cases}1, & if\ c > r \\ e^{1 - \frac{r}{c}}, & if\ c \leq r \end{cases} \\ 
BLEU = BP \times exp(\sum_{n=1}^{N}\omega_nlog\ p_n), log\ BLEU = min(1 - \frac{r}{c}, 0) + \sum_{n=1}^N\omega_nlog\ p_n$$

