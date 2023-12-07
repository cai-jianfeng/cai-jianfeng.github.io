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
通常包括 PCIE $\times 1 / 2 / 4 / 8 / 16$ 这 $5$ 种形式，并且 PCIE $\times 2x$ 的传输速率是 PCIE $\times x$ 的 $2$ 倍，其针脚数也增加 $2$ 倍(虽然 PCIE 接口/通道有多个针脚，但是每个针脚都是独立的串行传输)。
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

硬盘
===

<p style="text-align:justify; text-justify:inter-ideograph;"><b>机械硬盘(HDD)</b>使用 FAT 表(File Allocation Table，文件分配表)来描述文件系统内存储单元的分配状态及文件内容的前后链接关系。
由于它主要使用覆盖的方式写入数据，因此当对回收站进行清空操作时，只是删除 FAT 表中的记录，并没有直接删除机械硬盘对应位置的数据，等到下次系统重新写入数据到这个区域时，数据才会被覆盖。
而<b>固态硬盘(SSD)</b>无法使用覆盖方式写入数据，只能先将数据进行擦除，然后再写入。因此其内部有一个特殊的回收指令：<b>TRIM</b> 回收指令，它可以在监测到当前 SSD 未进行读写操作时，
对之前删除的区域进行擦除，则下一次系统重新写入数据到这个区域时，原始数据已经被擦除了，只需直接写入即可。注意，一般使用 SSD 时操作系统不使用 FAT 文件系统，而是其他更高级的文件系统。</p>

```windows
TRIM 状态查询命令：fsutil behavior query disabledeletenotify
TRIM 关闭命令：fsutil behavior set disabledeletenotify 1
TRIM 打开命令：fsutil behavior set disabledeletenotify 0
```

<p style="text-align:justify; text-justify:inter-ideograph;">机械硬盘有 LMR (Longitudinal Magnetic Recording)水平记录和 PMR (Perpendicular Magnetic Recording) 垂直记录；
PMR 中又包括 CMR (Conventional Magnetic Recording) 传统记录和 SMR (Shingled Magnetic Recording) 瓦叠记录：</p>

<ol><li><p style="text-align:justify; text-justify:inter-ideograph;">CMR：由于读写磁头是有大小的，所以磁道之间要给磁头预留空间，各个磁道之间存在空隙；
同时写磁头比读磁头大，间隙也是以大的写磁头为标准预留。这样磁道宽度够大，读磁头或写磁头经过某一磁道，都不会干扰到其他磁道。</p></li>
<li><p style="text-align:justify; text-justify:inter-ideograph;">SMR：为了提高密度，SMR 充分利用了磁道之间的空隙和读写磁头的相对大小，以较小的读磁头为标准来预留磁道宽度和间隙。
当读取数据时，读取磁头经过某一磁道，和其他磁道没有干扰。
但是写入数据时，由于写磁头更大且磁道宽度小，经过某一磁道时必定会覆盖下一个磁道的部分区域。
所以数据写入目标磁道前，会先标记受影响的磁道，把受到影响的数据暂存到别的地方(即机械硬盘缓冲区)，等目标磁道写入完成，再把数据读写回来。
而在写回暂存机械硬盘缓冲区的磁道数据时，又会影响下一磁道的数据，因此需要重复不断地暂存写回，直到到达最边缘的磁道。
为此，SMR 通常是以每个扇区为分界，扇区内使用 SMR 技术。</p></li></ol>