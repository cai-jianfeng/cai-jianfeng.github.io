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

<p style="text-align:justify; text-justify:inter-ideograph;">LaTeX 的文件名后缀为 ```.tex```。在编写 LaTeX 文件时，首先需要确定自己需要哪种文档类型，不同的文档类型的最终排版格式不同，同时在编写内容是使用的代码也有一定差异。</p>