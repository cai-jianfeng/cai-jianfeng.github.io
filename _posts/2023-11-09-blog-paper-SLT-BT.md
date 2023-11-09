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
假设原始数据集为 $D_{origin}^{1:M}$，输入手语视频为 $\mathbf{x} = \{x_t\}_{t=1}^T$，输出 sentence 为 $\mathbf{y} = \{y_u\}_{u=1}^U$，
而对应的 $glosses$ 为 $\mathbf{g} = \{g_v\}_{v=1}^V$。由于现实中拥有非常多的 sentence data，也叫 monolingual data。
本文便打算使用 back-translation 将 sentence 翻译为伪手语视频 $\hat{\mathbf{x}}$。
最直接的方法是先用现有的数据集 $D_{origin}^{1:M}$ 训练一个 text-to-sign 模型(输入 sentence，输出 sign video)，
然后使用额外收集的 sentence 集合 $\hat{\mathbf{y}}^j, j = [1,...,N]$ 输入到训练好的 text-to-sign 模型中，输出预测的手语视频 $\hat{\mathbf{x}}^j$。
这样生成的 $\{\hat{\hat{\mathbf{x}}^j,\mathbf{y}}^j\}$ 便可以作为伪数据集加入到原始数据集 $D_{origin}^{1:M}$ 一起训练一个 SLT 模型。
但是由于 sentence 和 video 的 modal gap，导致使用原始的小数据集训练出来的 text-to-sign 模型效果不好，继而影响生成的伪数据 $\hat{\hat{\mathbf{x}}^j$ 的质量。
因此，本文采用了两阶段的 back-translation 方法，在 text-to-gloss 阶段，由于 sentence 和 $glosses$ 的 modal 相似性，
本文直接训练了一个 text-to-gloss 的模型来将 sentence 翻译为 $glosses$；
而在 gloss-to-sign 阶段，由于仍然存在 $glosses$ 和 video 的 modal gap，
本文便放弃了使用模型直接生成的方式，而是充分采用了 $gloss$ 和 sign video 的单调性，使用分割的方式获得伪 video。
最终生成伪数据集 $\{\hat{\hat{\mathbf{x}}^j,\mathbf{y}}^j\}$。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">具体而言，如上图 (Figure 2)，在 <b>text-to-gloss</b> 阶段，
本文首先使用原始数据集 $D_{origin}^{1:M}$ 的 $\{\mathbf{y}^i, \mathbf{g}^i\}, i = [1,...,M]$ 训练了一个 text-to-gloss 的模型 $\boldsymbol{M}_{BT}(·)$。
然后使用大量和原始数据集 $D_{origin}$ 的 $\mathbf{y}^i$ 相同 domain 的其他 sentence(monolingual data，例如维基百科上的句子等) $\hat{\mathbf{y}}^j, j = [1,...,N]$ 
输入训练好的模型 $\boldsymbol{M}_{BT}(·)$ 中，
并输出预测的 $glosses\ \hat{\mathbf{g}}^j$，则生成的 $\{\hat{\mathbf{y}}^j, \hat{\mathbf{g}}^j\}$ 对就可以作为 gloss-to-text 的伪数据集。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">而在 <b>gloss-to-sign</b> 阶段，由于 $glosses$ 和 video 的 modal gap 仍然存在，因此不能直接训练模型生成伪数据。
但是，相比于 text-to-sign，$glosses$ 的好处是它和 sign video 具有单调性，手语视频中的每个动作都对应一个 $gloss$，且在时间顺序上和 $glosses$ 的序列顺序一致，
即假设 $x_{t_1:t_2}$ 对应 $g_i$，$x_{t_3:t_4}$ 对应 $g_j$，若 $i < j$，则 $t_1 < t_2 < t_3 < t_4$。
若是能够确定每个 $gloss\ g_i$ 所对应的 sign video 片段 $x_{t_l:t_r}$，则可以建立一个 gloss-to-sign 的对应表，将每个 $gloss_i$ 对应的 sign video 片段 $c_i$ 都存储起来。
那么对于 text-to-gloss 阶段生成的每一个 $\hat{\mathbf{g}}^j$，就可以按顺序将每一个 $\hat{g}_i^j$ 替换为表中相对应的 sign video 片段 $c_{i}^j$，这样就可以生成伪 sign video $\hat{\mathbf{v}}^j$。
因此，</p>