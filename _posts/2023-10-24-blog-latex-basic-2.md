---
title: 'latex 基础知识2'
date: 23-10-24
permalink: /posts/2023/10/blog-latex-basic-2/
tags:
  - 论文写作工具
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客延续<a href="https://cai-jianfeng.github.io/posts/2023/10/blog-latex-basic/" target="_blank" title="latex basic">
《latex基础知识》</a>，继续扩展关于 latex 的基本用法。</p>

Basic Application
===

<p style="text-align:justify; text-justify:inter-ideograph;">在一篇论文中，可能还会出现定理、推论、引理、定义等需要单独展示并使用特殊的展示样式，对于 latex 的不同文件格式其展示的样式也不同。
以定理为例，首先需要在导言区(即导包的区域)添加对定理的使用设置：</p>

<pre>
\newtheorem{theorem}{定理的显示前缀}[在何种条件下重新计数]
</pre>

<p style="text-align:justify; text-justify:inter-ideograph;">其中，\newtheorem{·}{·}[·] 的第一个 {} 中填写需要的格式，包括 theorem(定理)、corollary(推论)、lemma(引理)等；
而第二个 {} 中填写需要显示在定理前的前缀，并且每条定理也会自动编号。然后，在需要正文中需要插入定理的位置，使用如下代码函数生成定理：</p>

<pre>
\begin{theorem}
定理编写区
\end{theorem}
</pre>

<p style="text-align:justify; text-justify:inter-ideograph;">例如，我们需要编写一条定理 $1+1 = 2$，其前缀为 Theorem，这代码如下：</p>

<pre>
// 在导言区加入(注：latex 的注释是 %)
\newtheorem{theorem}{Theorem}
...
// 在需要添加定理的位置加入
\begin{theorem}
1 + 1 = 2
\end{theorem}
</pre>

