---
title: 'SLT-BT'
date: 23-11-09
permalink: /posts/2023/11/blog-paper-slt-bt/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://openaccess.thecvf.com/content/CVPR2021/html/Zhou_Improving_Sign_Language_Translation_With_Monolingual_Data_by_Sign_Back-Translation_CVPR_2021_paper.html" target="_blank" title="SLT-BT">Improving Sign Language Translation with Monolingual Data by Sign Back-Translation</a></p>

<p style="text-align:justify; text-justify:inter-ideograph;">发表会议：Conference on Computer Vision and Pattern Recognition (CVPR 2021)</p>

第一作者：Hao Zhou (USTC, Baidu VIS now)

Question
===

<p style="text-align:justify; text-justify:inter-ideograph;">如何解决 Sign Language Translation (SLT) 的数据缺乏问题</p>

Method
===

![SLT-BT architecture](/images/paper_SLT-BT.png)

<p style="text-align:justify; text-justify:inter-ideograph;">本文提出了使用 sign back-translation (SignBT) 的方式生成伪数据来解决数据缺乏问题。
具体而言，如上图 (Figure 2)，假设原始数据集为 $D_{origin}_{1:M}$，输入手语视频为 $\mathbi{x} = \{x_t\}_{t=1}^T$，输出 sentence 为 $\mathbi{y} = \{y_u\}_{u=1}^U$，
而对应的 $glosses$ 为 $\mathbi{g} = \{g_v\}_{v=1}^V$。
本文首先使用原始数据集 $D_{origin}_{1:M}$ 的 $\{\mathbi{y}^i, \mathbi{g}^i\}, i = [1,...,M]$ 训练了一个 sentence-to-gloss 的模型 $\boldsymbol{M}_{BT}(·)$。
然后使用大量和原始数据集 $D_{origin}$ 的 $\mathbi{y}^i$ 相同 domain 的其他 sentence(monolingual data，例如维基百科上的句子等) $\hat{\mathbi{y}}^j, j = [1,...,N]$ 
输入训练好的模型 $\boldsymbol{M}_{BT}(·)$ 中，
并输出预测的 $glosses\ \hat{\mathbi{g}}^j$，则 $\{\hat{\mathbi{y}}^j, \hat{\mathbi{g}}^j\}$ 对则可以作为生成的伪数据集。</p>