---
title: 'latex 问题合集'
date: 23-10-22
permalink: /posts/2023/10/blog-latex/
tags:
  - 论文写作工具
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要记录我在使用 latex 的过程中所遇到的问题和解决的方法(注：有些问题可能我自己也不知道原理，但是所有的解决方法都是亲测有效)。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">前言：我所使用的tex编辑器是 TeXstudio，它里面包含了多个 tex 编译器，包括 Latex、PDFLatex、XeLatex 等。</p>

- 遇到 **abstract** 和 **题目** 没有黑体字加粗，可能是编译器问题，选择 **pdflatex** 进行编译

- 在使用 .bib 文件引入 references 时，如何设置某些参考文献的字体为其他颜色：

```latex
% 在 .tex 文件开头引入包
\usepackage{xcolor}
\usepackage{xpatch}

% 最后引用 .bib 文件部分
% 中间这段 ==== 分割出来的部分看不懂，我是直接摘抄过来的
% ===================
\makeatletter
\ExplSyntaxOn
% #1 = color
% #2 = list of bib items
\cs_new:Npn \bibColoredItems #1#2
{
	\clist_map_inline:nn {#2} { \cs_new:cpn {bib@colored@##1} {#1} } 
}
\ExplSyntaxOff

% #1 = one bib item
\newcommand\bib@setcolor[1]{%
	\ifcsname bib@colored@#1\endcsname
	\expanded{\noexpand\color{\csname bib@colored@#1\endcsname}}%
	\else
	\normalcolor
	\fi
}

\if@tempswa
\xpatchcmd\@bibitem {\H@item}{\bib@setcolor{#1}\H@item}{}{\PatchFailed}
\xpatchcmd\@lbibitem{\H@item}{\bib@setcolor{#2}\H@item}{}{\PatchFailed}
\else
\xpatchcmd\@bibitem {\item}  {\bib@setcolor{#1}\item}  {}{\PatchFailed}
\xpatchcmd\@lbibitem{\item}  {\bib@setcolor{#2}\item}  {}{\PatchFailed}
\fi
\makeatother
% ===================

% 这里 需要改变颜色的参考文献 位置填写的内容和你在正文中 \cite{} 引用文献填写的内容一致
% 一般 .bib 文件中需要如下格式的引用文献：
% @ARTICLE{9479914,
% author={Nie, Wen and Huang, Kui and Yang, Jie and Li, Pingxiang},
% journal={IEEE Trans. Geosci. Remote Sens}, 
% title={A Deep Reinforcement Learning-Based Framework for PolSAR Imagery Classification}, 
% year={2022},
% volume={60},
% number={},
% pages={1-15},
% doi={10.1109/TGRS.2021.3093474}}
% 则 需要改变颜色的参考文献 位置填写的内容就是 9479914；
% 如果有多个参考文献需要设置颜色的话只需要 , 分割就行
\bibColoredItems{需要设置的颜色}{需要改变颜色的参考文献}
% 最后呈现的参考文献的格式(不如的模板格式不同), \bibliographystyle{} 中需要填写自己需要的格式（如IEEEtran 表示需要 IEEE transaction 的格式)
\bibliographystyle{指定的格式}
\nocite{*}
% 通常将所有参考文献都将它的 bibtex 格式的引用方式写在同一个文件中，并为这个文件命名为 名字.bib(如 reference.bib), 然后把文件名(如 reference)填写在 \bibliography{} 中
\bibliography{.bib文件的文件名}

```

