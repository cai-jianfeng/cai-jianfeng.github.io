---
title: 'The Basic Knowledge of TorchScript'
date: 23-12-22
permalink: /posts/2023/12/blog-torchscript/

[//]: # (star: superior)

tags:
  - 深度学习基本知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要讲解了使用 TorchScript 将 Python 模型代码转化为其他语言代码(如 C++)的原理和具体实现。
(<del>ps：到目前为止，我只了解了如何使用 TorchScript 将一个 PyTorch 模型转化为 TorchScript，并将其加载到 C++ 的代码中使用；
但是我并不了解 TorchScript 保存 PyTorch 模型的具体形式以及为什么 TorchScript 可以直接加载到 C++，
而传统的<code style="color: #B58900">torch.save</code>保存的 PyTorch 模型无法直接加载到 C++ 中，
其中应该和<b>序列化方法</b>有关联。我的计划是等我将其原理研究明白后再开始撰写这篇博客，但如果我迟迟无法理解，可能会先将 TorchScript 的具体使用先整理出来。
敬请期待⏳！</del>)</p>

<p style="text-align:justify; text-justify:inter-ideograph;">TorchScript 是一种 PyTorch 模型的中间表示，可以简单理解为一个 PyTorch 深度学习框架的子框架。
它拥有自己的 Torch Script 模型类(即不同于一般 PyTorch 模型类 <code style="color: #B58900">nn.Module</code>)，
并且可以使用 Torch Script 编译器对 Torch Script 模型进行理解、编译和序列化。最重要的一点是，Torch Script 模型可以在不同的语言上进行加载运行。
(这里可以类比于 Java 的 JVM，同一份 Java 可以在不同的操作系统上运行(Windows、Linux 等)，只需要对不同的平台编写不同的 JVM 即可。
而同一份 Torch Script 模型文件也可以在不同的语言上进行加载运行(Python、C++ 等)，
只需要对不同的语言编写不同的 Torch Script 操作包即可(Python 是 PyTorch 的 <code style="color: #B58900">torch.jit</code>，C++ 是 <code style="color: #B58900">LibTorch</code>)。)</p>

<p style="text-align:justify; text-justify:inter-ideograph;">这里有一个问题，即为什么需要 TorchScript 这种可以跨语言的模型格式存在？
顾名思义，PyTorch 的主要接口是 Python 编程语言。虽然对于许多需要动态性和易于迭代的场景来说，Python 是一种合适的首选语言。
但同样在许多情况下，Python 的这些属性是不利的。这些情况主要存在在生产环境中，其要求低延迟和严格部署要求。
对于生产场景，C++ 通常是首选语言，即使只是将其绑定到另一种语言，如 Java、Rust 或 Go。因此，在 Python 训练好模型后，要想部署到实际的应用场景中，通常需要使用 C++ 进行代码编写。
下面将概述 PyTorch 提供的转化方法，该方法从现有的 Python 模型转换为可完全在 C++ 中加载和执行的序列化表示，而不依赖于 Python。</p>

<p style="text-align:justify; text-justify:inter-ideograph;"><b>第一步：</b>需要将 PyTorch 模型转化为 Torch Script 模型。
PyTorch 提供了 $2$ 种方法可以将 PyTorch 模型转换为Torch Script。第一种是 <b>tracing</b> 机制，在这种机制中，模型的结构是通过使用示例输入对其进行一次 forward，并记录这些输入在模型中的流动情况来构建的。
这适用于控制流使用有限(即对<code style="color: #B58900">if/for/while</code>这种控制流代码的使用有限制)的模型。
第二种方法是向 PyTorch 模型中添加显式的注释(可以使用修饰器<code style="color: #B58900">@torch.jit.script</code>等)，
通知 Torch Script 编译器它可以直接解析和编译 PyTorch 模型代码，但要受 Torch Script 语言规则的约束。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">具体而言，</p>

References
===

1. [Loading a TorchScript Model in C++](https://pytorch.org/tutorials/advanced/cpp_export.html)

2. [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)

3. [TorchScript](https://pytorch.org/docs/stable/jit.html)