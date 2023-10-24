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

<p style="text-align:justify; text-justify:inter-ideograph;">在一篇论文中，可能还会出现<b>定理</b>、<b>推论</b>、<b>引理</b>、<b>定义</b>等需要单独展示并使用特殊的展示样式，对于 latex 的不同文件格式其展示的样式也不同。
以定理为例，首先需要在导言区(即导包的区域)添加对定理的使用设置：</p>

<pre>
\newtheorem{theorem}{定理的显示前缀}[在何种条件下重新计数]
</pre>

<p style="text-align:justify; text-justify:inter-ideograph;">其中，\newtheorem{·}{·}[·] 的第一个 {} 中填写需要的格式，包括 theorem(定理)、corollary(推论)、lemma(引理)、definition(定义)等；
而第二个 {} 中填写需要显示在定理前的前缀，并且每条定理也会自动编号；而第三个 [·] 中填写需要以谁为基准重新编号。然后，在需要正文中需要插入定理的位置，使用如下代码函数生成定理：</p>

<pre>
\begin{theorem}
定理编写区
\end{theorem}
</pre>

<p style="text-align:justify; text-justify:inter-ideograph;">例如，我们需要编写一条定理 $1+1 = 2$，其前缀为 Theorem，并且定理以 section 为基准重新编号，则代码如下：</p>

<pre>
% 在导言区加入(注：latex 的注释是 %)
% [section] 表示对于每个 section，定理都会重新自动编号
% 如在 section 1 中是 1.1, ...；则在 section 2 中就变成了 2.1, ...
% 这是 latex 的自动编号，无需我们手动编写
% 而 {Theorem} 表示会生成前缀，如对 section 1 的第一条定理，会生成前缀 Theorem 1.1
\newtheorem{theorem}{Theorem}[section]
...
% 在需要添加定理的位置加入
\begin{theorem}[定理别名]
% 在定理编写区，数学符号用 \(·\) 包裹展示，而数学表达式用 \[·\] 包裹展示(其语法和正文的 $·$ 中的语法相同)
This is a theorem about \(f\) function: \[ 1 + 1 = 2 \]
\end{theorem}

最终生成的效果是：
Theorem 1.1 (定理别名): This is a theorem about f function: 
                    1 + 1 = 2
</pre>

<p style="text-align:justify; text-justify:inter-ideograph;">此外，<b>列表</b>也是论文中可能会出现的类型，LaTeX 中的列表环境包含无序列表 itemize、有序列表 enumerate 和描述 description，它们都是使用如下代码函数进行编写：</p>

<pre>
\begin{列表类型(包括 itemize/enumerate/description)}
    % 当使用 [自定义列表标号] 后，默认的列表标记符号就不会显示
    \item[自定义列表标号] ...;
    \item[...] ...;
    ...
\end{列表类型}
</pre>

<p style="text-align:justify; text-justify:inter-ideograph;">上一篇 blog 提到了如何编写显示数学公式的代码函数，但对于数学公式本身的编写的语法没有提及。这里对一些基础的公式进行说明：</p>

<p style="text-align:justify; text-justify:inter-ideograph;">首先是 <b>上下标</b>，使用 _{} / ^{} 输入(在上下标只有一个字符时 {} 可以省略)，如 a_n / a^n 分别表示 $a_n / a^n$。
其次是<b>分式</b>，使用 \dfrac{}{} / \frac{}{} 输入(区别在于后者的字体较小，一般用于指数等地方)，如 \dfrac{a}{b} / \frac{a}{b} 分别表示 $\dfrac{a}{b} / \frac{a}{b}$。
</p>