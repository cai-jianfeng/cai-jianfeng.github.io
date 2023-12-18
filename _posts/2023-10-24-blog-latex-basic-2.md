---
title: 'The Basic Knowledge Of Latex 2'
date: 23-10-24
permalink: /posts/2023/10/blog-latex-basic-2/
star: superior
tags:
  - 论文写作工具
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客延续 <a href="https://cai-jianfeng.github.io/posts/2023/10/blog-latex-basic/" target="_blank" title="latex basic">
The Basic Knowledge Of Latex</a>，继续扩展关于 latex 的基本用法。</p>

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

<p style="text-align:justify; text-justify:inter-ideograph;">上一篇 blog 提到了如何编写显示数学公式的代码函数，但对于数学公式本身的编写的语法没有提及。这里对一些基础的公式进行说明(注意，下述的代码都是要写在上篇 blog 提到的公式代码函数所包裹的区域内的)：</p>

<p style="text-align:justify; text-justify:inter-ideograph;">首先是 <b>上下标</b>，使用 _{} / ^{} 输入(在上下标只有一个字符时 {} 可以省略)，如 a_n / a^n 分别表示 $a_n / a^n$。
其次是<b>分式</b>，使用 \dfrac{}{} / \frac{}{} 输入(区别在于后者的字体较小，一般用于指数等地方)，如 \dfrac{a}{b} / \frac{a}{b} 分别表示 $\dfrac{a}{b} / \frac{a}{b}$。
然后是括号，对于花括号 {}，由于默认的情况表示范围，所以一般不会显示，需要显示则需要使用转义符 \，即 \{\} 显示为 $\{\}$，
而对于需要较大的括号，则需要使用 \left(...\right)，其显示为 $\left(...\right)$。此外，在中间需要隔开时，可以用 \left(..\middle|..\right)，其显示为 $\left(..\middle|..\right)$。
除了常规括号，还有<b>单边大括号</b>，其使用如下代码函数进行输入：</p>

<pre>
\begin{cases}
    % \\ 表示换行，即各行使用 \\ 分隔
    % 同时，case 以 & 符号作为对齐标志，即对齐每行的 & 的左右两边
    equation1 & condition1 ... \\
    equation2 & condition2 ...
\end{cases}
</pre>

<p style="text-align:justify; text-justify:inter-ideograph;">接着是<b>求和/求积公式</b>，分别使用 \sum / \prod 输入(上下标表示和上述相同)，如 \sum_i^N{a_i} / \prod_j^M{b_j} 分别表示 $\sum_i^N{a_i} / \prod_j^M{b_j}$。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">然后是<b>矩阵和行列式</b>，分别使用如下的代码函数进行输入：</p>

<pre>
% bmatrix/pmatrix/vmatirx 分别表示 方括号矩阵、圆括号矩阵、行列式
\begin{bmatrix/pmatrix/vmatirx}
    % 各个元素使用 & 间隔，各行使用 \\ 分隔
    a & b \\
    c & d
\end{bmatrix}
</pre>

<p style="text-align:justify; text-justify:inter-ideograph;">另外，公式中可能对于某些字符需要<b>加粗显示</b>，这里建议使用 \bm{} 代码函数(需要导入 bm 宏包 \usepackage{bm})进行加粗，可以保留它的斜体属性。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">最后，在论文中可能会出现公式推导的情况，这时候需要对每个公式的等号进行<b>对齐</b>，此时可以使用如下代码函数进行输入：</p>

<pre>
\begin{aligned}
% aligned 以 & 符号作为对齐标志，即对齐每行的 & 的左右两边
    a & =b+c \\
    & =d+e
\end{aligned}
</pre>

<p style="text-align:justify; text-justify:inter-ideograph;">在公式中的换行符是 \\，但是在 \begin{equation} 的代码环境下并不起作用，这时如果因为公式过长或者其他原因需要强制换行，
可以放弃使用 \begin{equation}，使用 \begin{multline} (需要导入 amsmath 宏包 \usepackage{amsmath})：</p>

<pre>
\begin{multline}
    % 注意，默认第一行的公式是靠左对齐，最后一行的公式是靠右对齐
    ....
\end{multline}
</pre>

Code Demo
===
<p style="text-align:justify; text-justify:inter-ideograph;">最后，对上面所有的内容集成到一个 `test2.tex` 文件中，并展示其编译后的 pdf 文件内容形式。其中， `test2.tex` 文件中的内容如下：</p>

<pre>
\documentclass[lettersize,journal]{IEEEtran}
\usepackage{bm}
\usepackage{amsmath}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{definition}{Definition}[section]
\begin{document}
	\title{Latex basic knowledge 2}
	\author{Jianfeng Cai}
	
	\maketitle
	
	\section{Introduction}
	
	\begin{theorem}[Pythagorean theorem]
		\label{pythagorean}
		This is a theorem about right triangles and can be summarised in the next 
		equation 
		\[ x^2 + y^2 = z^2 \]
	\end{theorem}
	
	\begin{description}
		\item[(1)] first; 
		\item[(2)] second;
		\item[(3)] third. 
	\end{description}

	$$
	a_i^n + b_j^m = 0
	$$
	$$
	a^{\dfrac{num}{den}} / a^{\frac{num}{den}}
	$$
	$$
	\left(1 + \frac{1}{2} \middle | \bm{\{i \neq j\}}\right)
	$$
	\begin{equation}
		f = \begin{cases}
			equation1 & condition1 ... \\
			equation2 & condition2 ...
		\end{cases}
	\end{equation}
	$$
	\begin{bmatrix}
		a & b \\
		c & d
	\end{bmatrix}
	\begin{pmatrix}
		a & b \\
		c & d
	\end{pmatrix}
	\begin{vmatrix}
		a & b \\
		c & d
	\end{vmatrix}
	$$
	$$
	\begin{aligned}
		a & =b+c \\
		& =d+e
	\end{aligned}
	$$
	\begin{multline}
		f(x) = \\
		\prod_{min}^{max} b\\
		\sum_{min}^{max} c \\
		= 0
	\end{multline}	
\end{document}
</pre>

<p style="text-align:justify; text-justify:inter-ideograph;">编译器使用 PDFLaTeX，最终编译生成的 PDF 文件内容如下：</p>

![demo](/images/latex_basic_application2.png)