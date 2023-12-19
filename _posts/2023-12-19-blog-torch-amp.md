---
title: 'The Basic Knowledge of Automatic Mixed Precision'
date: 23-12-19
permalink: /posts/2023/12/blog-torch-amp/
star: superior
tags:
  - 深度学习基本知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要讲解了使用自动混合精度(AMP)降低模型内存占用的原理和具体实现。</p>

Automatic Mixed Precision
===

<p style="text-align:justify; text-justify:inter-ideograph;">混合精度训练(MPT)主要是指在模型训练过程中，将其中一些操作的输入输出数据使用 $float16$ 的半精度数据类型存储，无法使用的则正常使用 $float32$ 单精度数据类型存储。
而自动混合精度(AMP)则是自动地执行 MPT，即自动给每个操作的输入输出数据匹配合适的数据类型。通常而言，<b>AMP</b> 主要包括 $2$ 个部分：<b>autocast</b> 和 <b>grad scale</b>。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">其中，autocast 即狭义上的 AMP。在一般的模型训练中，PyTorch 对数据存储都是采用 $float32$ 的单精度形式，则对于每个数据的存储都需要使用 $32$ 位的空间。
反观 $float16$，它的内存占用仅为 $float32$ 的一半，这使得在使用 $float16$ 的情况下，同等内存容量的 GPU/CPU 可以训练更大的模型、使用更大的 batch size。
同时，$float16$ 在计算上由于经过硬件优化，计算速度一般会快于 $float32$，并且由于数据存储位数减少，其在 DDP / MP 等训练时的通信量也减少，即减少等待时间，加快数据的流通和模型的训练。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">但是，使用 $float16$ 存在一些问题。首先是<b>数据溢出问题(overflow/underflow)</b>。
由于 $float16$ 的表示范围和精度 $ll$ $float32$，因此很可能出现数据超过其表示范围(overflow)或者小于其表示范围(underflow)。
其中后者在训练时更为常见(在训练后期模型参数的梯度很小，很有可能小于 $float16$ 的数据范围)。
其次是<b>舍入误差(Rounding Error)</b>，在 backward 计算完成梯度后，需要进行参数更新：<code style="color: #B58900">p = p - lr * p.grad</code>。
此时由于参数的数量级 $gg$ 梯度的数量级，导致在参数的数量级下的 $float16$ 最小间隔数量级增加，进而导致即使梯度大于 $float16$ 的最小表示，也会在与参数相加后被舍弃，从而导致本次参数更新失败。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">为了解决第二个问题，autocast 采用 $float16$ 和 $float23$ 数据精度混合的方式来表示不同的数据，并备份参数的 $float32$ 版本来进行参数更新。
具体而言，首先备份一份参数的 $float32$ 版本，然后在模型的 forward 过程中，权重(weights)和数据(datas)都使用 $float16$ 表示，则模型计算的中间输出结果(activations)也使用 $float16$ 表示。
而在模型的 backward 过程中，由于中间输出结果(activations)和权重(weights)都是使用 $float16$ 表示，则计算得到的各自的梯度(activation grad 和 weight grad)也是使用 float16$。
其中 activation grad 主要是为了参与位于其后面的模型参数梯度的计算，因此只需保持 $float16$ 即可；而 weight grad 需要进行模型参数的更新。
为了防止舍入误差，需要将 weight grad 转化为 $float32$ 的数据表示以提高数据表示范围，接着与备份的参数的 $float32$ 版本进行参数更新。由于将数据表示都提高到了 $float32$，因此不会出现舍入误差问题。
上述的具体数据表示如下图所示，可以看到将模型的大部分数据都使用 $float16$ 进行表示，节省了大量的内存空间。

![AMP principal](/images/AMP_principal.png)

<p style="text-align:justify; text-justify:inter-ideograph;"></p>