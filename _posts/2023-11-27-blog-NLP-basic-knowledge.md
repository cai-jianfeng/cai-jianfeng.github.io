---
title: 'The Basic Knowledge of NLP'
data: 23-11-27
permalink: '/posts/2023/11/blog-NLP-basic-knowledge'
tags:
  - 深度学习基础知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要介绍了 NLP 任务中的基础知识，包括性能评价指标等。</p>

<h1>Metric</h1>

<p style="text-align:justify; text-justify:inter-ideograph;"><b><a href="https://aclanthology.org/P02-1040.pdf" target="_blank">BLEU (BLEU Score)</a></b>：Bilingual Evaluation Understudy，
主要用于计算一对多的 Translation 的任务的质量，例如  Machine Translation。这种任务通常拥有多个 ground-truth (<a href="https://github.com/cai-jianfeng/glossification_editing_programs/blob/main/bleu_score.py" target="_blank">参考代码</a>)。其计算公式如下：</p>

$$BLEU_{n-gram} = BP \times exp(log\ P_n),\ BLEU = BP \times exp(\dfrac{\sum_{i=1}^N\omega_nlog\ P_n}{N}), \\ 
P_n = \dfrac{\sum_{n-gram \in \hat{y}}{Counter_{Clip}(n-gram)}}{\sum_{n-gram \in \hat{y}}{Counter(n-gram)}}, 
BP = \begin{cases}1, & L_{out} > L_{ref} \\ exp(1 - \dfrac{L_{ref}}{L_{out}}), & L_{out} \leq L_{ref} \end{cases}$$

<p style="text-align:justify; text-justify:inter-ideograph;">其中，$\omega_n$ 表示 $n-gram$ 的权重；$Out$ 表示预测的句子 $\hat{y}$，$Ref$ 表示 ground-truth 的句子 $y$；$n-gram$ 表示由 $n$ 个词组成的词组；
$Counter_{Clip}(x)$ 表示 $x$ 在 $Ref$ 中出现的次数和在 $Out$ 中出现的次数的最小值；
$Counter(x)$ 表示 $x$ 在 $Out$ 中出现的次数；$L_{out/ref}$ 表示 $Out/Ref$ 的长度。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">对于整个测试集 $\mathcal{D}_t:\{S_i,y_i\}_{i=1}^{M}$，BLEU 的计算公式如下：</p>

$$p_n = \dfrac{\underset{\mathcal{C} \in \{Candidates\}}{\sum}\underset{n-gram \in \mathcal{C}}{\sum} Count_{clip}(n-gram)}{\underset{\mathcal{C'} \in \{Candidates\}}{\sum}\underset{n-gram' \in \mathcal{C'}}{\sum} Count(n-gram')}, BP = \begin{cases}1, & if\ c > r \\ e^{1 - \frac{r}{c}}, & if\ c \leq r \end{cases} \\ 
BLEU = BP \times exp(\sum_{n=1}^{N}\omega_nlog\ p_n), log\ BLEU = min(1 - \frac{r}{c}, 0) + \sum_{n=1}^N\omega_nlog\ p_n$$

<p style="text-align:justify; text-justify:inter-ideograph;">其中，$\{Candidates\} = \{\hat{y}_i\}_{i=1}^M$；
$c$ 表示 $\mathcal{D}_t$ 中所有预测的句子 $\hat{y}_i$ 的长度 $L_{out}^i$ 的总和：$c = \sum_{i=1}^ML_{out}^i$，
$r$ 表示 $\mathcal{D}_t$ 中所有 ground-truth 的句子 $y_i$ 的长度 $L_{ref}^i$ 的总和：$r = \sum_{i=1}^ML_{ref}^i$ 
(如果每个 $S_i$ 对应不止一个目标句子 $y_i^j, j=1,...,J$，
则 $L_{ref}^i$ 表示与预测的句子 $\hat{y}_i$ 的长度最短的目标句子 $y_i^j$ 的长度 $L_{ref}^{i,j}$：$\underset{y_i^j, j = [1,...,J]}{arg\ min}{|L_{out}^i - L_{ref}^{i,j}|_1}$)。
<b><span style="color: red">BlEU Score 是通过将整个测试/训练/验证集的文本作为一个整体来计算的。因此，不能对集合中的每个句子单独计算 BlEU Score，然后用某种方式平均得到最终的分数。</span></b></p>

<p style="text-align:justify; text-justify:inter-ideograph;"><b>WER</b>：Word Error Rate，主要用于计算一对一的 Translation 的任务的质量，例如 Speech-to-text。这种任务通常只有一个 ground-truth。其计算公式如下：</p>

$$WER = \dfrac{d_{Levenstein}}{L_{out}} = \dfrac{d_{insert} + d_{delete} + d_{substitute}}{L_{out}}$$

<p style="text-align:justify; text-justify:inter-ideograph;">其中，$L_{out}$ 表示预测的句子 $\hat{y}$ 的长度；$d_{Levenstein}$ 表示 Levenstein distance，即编辑距离，表示将预测的句子 $\hat{y}$ 转化为 ground-truth $y$ 所需的编辑步骤。
而所定义的编辑步骤由 $3$ 个部分组成：<b>Insert(i)</b>，表示在第 $i$ 个位置添加一个词；<b>Delete(i)</b>：表示删除第 $i$ 个位置的词；
<b>Substitute(i, w)</b>：表示将第 $i$ 个位置上的词换成 $w$。$d_{insert/delete/substitute}$ 分别表示将预测的句子 $\hat{y}$ 转化为 $y$ 所需的 $insert/delete/substitute$ 的次数。
通常情况下，我们不会分开计算 $3$ 个编辑步骤的各自次数，
而是使用 <b><a href="https://github.com/cai-jianfeng/glossification_editing_programs/blob/main/extract_editoring_program.py" target="_blank">DP</a></b> (Dynamic Programming，
动态规划)算法直接求解 $d_{Levenstein}$。</p>