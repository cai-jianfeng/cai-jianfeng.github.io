---
title: 'The Basic Knowledge of Computer Hardware'
data: 23-12-06
permalink: '/post/2023/12/blog-hardware'
tags:
  - 电脑硬件基本知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要介绍了电脑硬件中的基础知识(ps；强烈安利 B 站硬件茶谈的<a href="https://space.bilibili.com/14871346/channel/collectiondetail?sid=550815" target="_blank">硬件科普视频</a>，讲的太好了🙂。虽然他现在恰饭有点多😥)</p>

CPU
===

<p style="text-align:justify; text-justify:inter-ideograph;"><b>OEM vs ODM</b>：OEM 即原始设备制造商，是指根据厂商的设计需求进行加工生产的制造商；ODM 即原始设计制造商，是指根据厂商的需求进行设计的设计商。
因此，一个知名的 CPU 品牌，可能是该公司委托某 ODM 进行设计，然后再经过某 OEM 进行加工生产，等到拿到成品 CPU 后只需贴上自己的公司品牌即可销售。</p>

<p style="text-align:justify; text-justify:inter-ideograph;"><b>散片 vs 盒装</b>：散片就是 OEM 厂商供应给 CPU 品牌公司的批发 CPU；而盒装则是 CPU 品牌公司专门供应给普通用户的零售 CPU。</p>

频率
===

<p style="text-align:justify; text-justify:inter-ideograph;"><b>晶振</b>：晶体振荡器(crystal oscillator，简称 XO)，是主板上提供 $100$Mhz 基础频率的部件，来确保所有设备协调同步。</p>

<p style="text-align:justify; text-justify:inter-ideograph;"><b>外频</b>：是设备 $A$ 与其他设备进行交互时的频率，以确保与其他设备交互的同步，其值等于晶振频率；
<b>主频</b>：设备 $A$ 在进行自我工作时的频率，以确保充分发挥设备性能；<b>倍频</b>：等于主频 / 外频。
通常 CPU 和内存本身的频率较高，如果仅仅只在晶振频率上进行工作无法有效发挥自身性能，而其他设备频率较低，在与它们进行交互时又只能使用晶振频率，
因此在与其他设备交互时，CPU 和内存通常使用外频，而在自身工作时，通常使用主频。
此外，内存有种机制称为<b>内存外频异步工作</b>，可以使得内存外频 CPU 外频高 $33.3\dot{3}$ 的频率下(即 $133.3\dot{3}$)运行，因此内存拥有 $2133$、$2666$ 的非整百倍频率。
在具体操作中可以在 BIOS 中调节内存速度转化比率 Ratio 为 $100:100 / 100:133$。
但是显卡核心频率和显存频率不受晶振频率限制，可以随意设置，因此它们是通过 PCIE 接口和 CPU 相连，并不是通过主板的 PCIE 总线。</p>

总线
===

<p style="text-align:justify; text-justify:inter-ideograph;"><b>PCIE 总线</b>：串行总线，包含 PCIE 接口(如显卡接口)和 PCIE 通道(如 M.2 接口和雷电 3 接口)，
通常包括 PCIE $\times 1 / 2 / 4 / 8 / 16$ 这 $5$ 种形式，并且 PCIE $\times 2x$ 的传输速率是 PCIE $\times x$ 的 $2$ 倍，其针脚数也会增加(虽然 PCIE 接口/通道有多个针脚，但是每个针脚都是独立的串行传输)。
下表展示不同版本的 PCIE 接口的传输速率。(小 tips：残血雷电 3 / M.2 表示 PCIE $\times 2$；满血雷电 3 / M.2 表示 PCIE $\times 4$。)</p>

|           | $\times 1$ | $\times 2$ | $\times 4$ | $\times 8$ | $\times 16$ |
|:---------:|:----------:|:----------:|:----------:|:----------:|:-----------:|
| PCI-e 1.0 | $250$MB/s  | $500$MB/s  |  $1$GB/s   |  $2$GB/s   |   $4$GB/s   |
| PCI-e 2.0 | $500$MB/s  |  $1$GB/s   |  $2$GB/s   |  $4$GB/s   |   $8$GB/s   |
| PCI-e 3.0 |  $1$GB/s   |  $2$GB/s   |  $4$GB/s   |  $8$GB/s   |  $16$GB/s   |
| PCI-e 4.0 |  $2$GB/s   |  $4$GB/s   |  $8$GB/s   |  $16$GB/s  |  $32$GB/s   |

<p style="text-align:justify; text-justify:inter-ideograph;"><b>南桥芯片组</b>：在主板上，只有 PCIE 接口和内存是直接和 CPU 的 PCIE 处理器相连接；
而其他的设备(如硬盘等)都是通过 PCIE 通道和南桥芯片组相连，再由南桥芯片组和 CPU 通过 PCIE 通道相连。下图展示 Intel 和 AMD 的总线布局：</p>

![South Bridge Chipset](/images/hardware_South_Bridge_Chipset.png)