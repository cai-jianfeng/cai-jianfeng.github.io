---
title: 'latex 基础知识'
date: 23-10-22
permalink: /posts/2023/10/blog-latex-basic/
tags:
  - 论文写作工具
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要记录了 latex 的基本知识和用法，适用于第一次听到 latex 这个写作排版工具的小白。</p>

What is latex?
===

<p style="text-align:justify; text-justify:inter-ideograph;">首先需要了解 TeX，它是一个为排版文字和数学公式而开发的软件，但是它的使用操作复杂，
而 LaTeX 是在 TeX 基础上开发的 TeX 扩展命令集，相当于对 TeX 的进一步封装，通过整合常用的版面设置操作，降低排版的工作量和难度。
它们之间的关系类似于 汇编语言 和 C语言的关系：C语言是对汇编语言的进一步封装，使得人们可以更加方便地使用。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">与其他的写作工具(如 Word)不同，LaTeX 是通过输入代码命令(简称代码)来完成一系列的操作的，如公式插入，图片插入，排版，脚注，引用，参考文献等。
因此，它的写作自由度较低，例如输入错误一条代码就会导致这个文件编译错误；在排版时无法提前获知每张图片/表格的具体位置(哪怕你在其中加入了限制代码)等。
但是它的限制性也给它带来了极强的规范性。例如标题、章节、图表、参考文献等自动标号及引用它们的方便性；公式输入的便捷性和多样性等。
因此它非常适用于论文的写作排版，是科研人员必不可少的写作利器。</p>

Install
===

<p style="text-align:justify; text-justify:inter-ideograph;">既然 LaTeX 是  TeX 的封装，其内部的代码还是需要 TeX 工具来解析编译，因此首先我们需要安装 TeX 的编译器：<a href="https://tug.org/texlive/" target="_blank" title="TeX Live">TeX Live</a>。
它是 TUG (TeX User Group) 维护和发布的 TeX 系统，可以说是「官方」的 TeX 系统，对于 TeX 的维护和开发是最频繁的。
其中包含的多个 TeX 编译器，如 Latex、PDFLatex、XeLatex 等。
因此推荐任何阶段的 TeX/LaTeX 用户，都尽可能使用 TeX Live，以保持在跨操作系统平台、跨用户的一致性。
其次我们需要一个编辑器来编写我们的 LaTeX 代码，常见的编辑器有 TeXworks、<a href="https://texstudio.sourceforge.net/" target="_blank" title="TeXstudio">TeXstudio</a>、VSCode等。
我使用的是 TeXstudio，在使用的这段时间感觉还是不错的，它的编译器选择在工具栏的 tool - commands。</p>

Basic Application
===

<p style="text-align:justify; text-justify:inter-ideograph;">LaTeX 的文件名后缀为 `.tex`。在编写 LaTeX 文件时，首先需要确定自己需要哪种文档类型(必要的)，不同的文档类型的最终排版格式不同，同时在编写内容是使用的代码也有一定差异。
在 LaTeX 中，大部分的代码格式都是 \代码函数名[基本参数]{内容}，在无需设置特定的基本参数(即全部使用默认)时可以不写 [基本参数]，
例如，选择 IEEEtran 的文档格式，并且基本参数为 lettersize,journal，则可以使用代码 \documentclass[lettersize,journal]{IEEEtran}。
接着是引入编写内容时所需要用到的宏包(类似于 C 语言的 include 和 python 的 import)来引入自己所要使用的代码函数，引入宏包的代码函数为 \usepackage{包名}，
例如，你想在文章中使用 \cite{} 这个代码函数来对参考文献进行引用，就需要在这里引入宏包 \usepackage{cite}。
接下来便是正文部分，它使用代码函数：</p>

<center><pre>
\begin{document}
...
\end{document}
</pre></center>

<p style="text-align:justify; text-justify:inter-ideograph;">对内容进行包裹。所有需要在论文中展示出来的部分都必须写在这里面。
一篇论文中主要包括了标题、作者、章节、图表、公式、参考文献。下面将一一讲解它们的使用方式。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">首先是标题，它使用代码函数 \title{标题}，而作者使用 \author{作者名}。
在代码函数填写的内容中(即 {} 里的内容)，如果有多个需要填写，一般使用 , 分隔，例如对于多个作者，则使用 \author{作者名1, 作者名2, ...}。
为了能使标题和作者显示在论文中，还需要使用 \maketitle 来进行显示设置。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">其次是章节，即目录，根据不同的目录等级使用不同的代码函数。首级的目录使用 \section{目录名}，次级目录使用 \subsection{目录名}，三级目录使用 \subsubsection{目录名}，
‘IEEETran’ 格式的文件最多只支持三级目录。除此之外，还可以给每个目录名设置标签以方便我们在正文中进行引用，具体代码函数为 \label{标签名}，需要紧接在目录代码函数之后。
在设置完标签后，我们便可以使用标签来对目录进行引用，例如，我们设置目录 A 的标签为 la，则我们在正文中就可以使用 \cite{la} 对其进行引用：</p>

<center><pre>
\section{A}
\label{la}
...
\cite{la}
</pre></center>

<p style="text-align:justify; text-justify:inter-ideograph;">接着是图表，想要在 LaTeX 中插入图片，需要引入宏包 \usepackage{graphicx}，接着便可使用如下代码函数进行图片插入：</p>

<center><pre>
\begin{figure}  -插入图片代码函数: figure 表示插入一栏(对于两栏的论文而言), figure* 表示插入两栏
	\centering  -表示居中对齐
	\includegraphics[width=3.5in]{path/to/figure}  -插入图片: width 表示设置宽度，除了具体数值，还可以设置为 \textwidth等 LaTeX 自带的常数变量; path/to/figure 表示图片位置(相对 .tex 文件的位置)
	\caption{图片标题}
	\label{标签名}  -设置标签，在正文中就可以使用 \cite{标签名}进行图片引用
\end{figure}
</pre></center>

<p style="text-align:justify; text-justify:inter-ideograph;">需要注意的是，LaTeX会对每一个插入的图片按插入顺序进行自动编号 Fig. 1., Fig. 2. ,...，因此无需自己在图片标题中标号。
而想要在 LaTeX 中插入表格，可以使用如下代码函数(同样，LaTeX 也会对插入的所有表格按顺序进行自动编号)：</p>

<center><pre>
\begin{table}
	\caption{表格标题}
	\label{标签名}
	\begin{tabular}{ccc}  - 表格内容: {ccc} 表示一共有多少列, 下面每行内容的列数都必须与它保持一致
		1 & 2 & 3 \\  - 每一行表格内容: 每一列用 & 分割, 使用 \\ 换行(表示后面是下一行的内容)
		4 & 5 & 6
	\end{tabular}
\end{table}
</pre></center>

<p style="text-align:justify; text-justify:inter-ideograph;">然后是公式，公式的编写格式和 markdown 非常相似，其中：</p>

<center><pre>
$
公式内容
$
</pre></center>

<p style="text-align:justify; text-justify:inter-ideograph;">表示嵌入在段落中的公式，即它和前段文字与后段文字都在同一行中。而</p>

<center><pre>
$$
公式内容
$$
</pre></center>

<p style="text-align:justify; text-justify:inter-ideograph;">表示独立公式，即它会与前段文字与后段文字分离。成为单独的一个公式展示在一行。
所以第一种方式适用于嵌入在行中的公式，而第二种方式适用于一些大型公式或者较为重要的公式，需要使用单独行进行展示。
对于第二个问题，它有一个问题，就是公式后面没有标号，无法对其进行引用。因此，在论文写作中，基本上不会使用该方式，而是采用另一种方式，即使用代码函数：</p>

<center><pre>
\begin{equation}
	a + b = c
\end{equation}
</pre></center>

<p style="text-align:justify; text-justify:inter-ideograph;">这样生成的公式不仅单独展示在一行，还有 LaTeX 的自动编号。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">最后是参考文献，这里建议使用 `.bib` 文件来实现参考文献的引入。
具体而言，首先在和 `.tex` 文件相同的位置创建一个 `.bib` 文件，然后对于每一篇你想引用的文献，找到它的 bibtex 引用格式(在 google scholar 上可以找到)，并将其按顺序复制到 `.bib` 文件中。
然后在 `.tex` 文件的正文末尾(\end{document}之前)，使用如下代码函数引入参考文献：</p>

<center><pre>
\bibliographystyle{参考文献格式}  -和 \documentclass 一样，不同的参考文献格式所展示的参考文献分格不同
\nocite{*}  -是否把 .bib 文件中的所有参考文献都展示: 如果没有该代码函数，LaTeX 只会展示出在正文中被引用过的参考文献
\bibliography{.bib 文件名}  -.bib 文件的文件名
</pre></center>

<p style="text-align:justify; text-justify:inter-ideograph;">除了在最后列出参考文献，我们还需要在正文中引用参考文献，一般而言，参考文献的 bibtex 引用格式(即复制在 `.bib` 文件中的格式)为：</p>

<center><pre>
@article{参考文献缩写,
  title={...},
  author={...},
  ...
}
</pre></center>

<p style="text-align:justify; text-justify:inter-ideograph;">此时，我们只需要在正文中需要引用的位置使用代码函数 \cite{参考文献缩写} 即可引用该参考文献。</p>