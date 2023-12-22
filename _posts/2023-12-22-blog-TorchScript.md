---
title: 'The Basic Knowledge of TorchScript'
date: 23-12-22
permalink: /posts/2023/12/blog-torchscript/

[//]: # (star: superior)

tags:
  - 深度学习基本知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要讲解了使用 TorchScript 将 Python 模型代码转化为其他语言代码(如 C++)的原理和具体实现。
(ps：到目前为止，我只了解了如何使用 TorchScript 将一个 PyTorch 模型转化为 TorchScript，并将其加载到 C++ 的代码中使用；
但是我并不了解 TorchScript 保存 PyTorch 模型的具体形式以及为什么 TorchScript 可以直接加载到 C++，
而传统的<code style="color: #B58900">torch.save</code>保存的 PyTorch 模型无法直接加载到 C++ 中，
其中应该和<b>序列化方法</b>有关联。我的计划是等我将其原理研究明白后再开始撰写这篇博客，但如果我迟迟无法理解，可能会先将 TorchScript 的具体使用先整理出来。
敬请期待⏳！)</p>

References
===

1. [Loading a TorchScript Model in C++](https://pytorch.org/tutorials/advanced/cpp_export.html)

2. [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)

3. [TorchScript](https://pytorch.org/docs/stable/jit.html)