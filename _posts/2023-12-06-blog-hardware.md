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

<p style="text-align:justify; text-justify:inter-ideograph;">固态硬盘(SSD)主要由浮栅晶体管堆叠而成，浮栅晶体管包括控制级、$P$ 级、浮栅级、源级和漏级，具体结构如下图，其中浮栅级中存储一定量电子。
对于读取数据，可以在源级和漏级之间施加电压：如果电路导通，则说明当前浮栅中存在大量电子，该状态表示为 $0$；如果没有导通，则说明当前浮栅中存在少量电子，该状态表示为 $1$。
而对于写入数据，向 $P$ 级施加电压可以从浮栅中析出电子，表示写入 $1$；向控制级可以令浮栅吸回电子，表示写入 $0$。</p>

<img src="https://cai-jianfeng.github.io/images/hardware_SSD.png">

<p style="text-align:justify; text-justify:inter-ideograph;">因此，由浮栅晶体管堆叠而成的区域称为 <b>NAND</b>。除此之外，SSD 一般还包括一个主控和一个缓存，其中主控主要控制数据的读写以及区域分配调度；缓存主要暂存需要写入 NAND 的数据/需要读入内存的数据。
一般主控先使用缓存进行交互，当缓存写入已满/读取的数据不在缓存，才与 NAND 进行交互(缓存的读写速度 $\gg$ NAND)。</p>

<p style="text-align:justify; text-justify:inter-ideograph;"><b>磁盘阵列</b>：RAID，是一种将多个磁盘进行组合来存储数据以实现数据保护的方法(这里的数据保护是指对数据有容错能力，当部分数据损毁时也可以恢复出完整数据)。
其分为 $0 \sim 7$ 中方式：</p>

<ol><li>
<p style="text-align:justify; text-justify:inter-ideograph;">RAID $0$：假设有 $X$ 块磁盘，RAID $0$ 通过将需要存储的数据平均分成 $X$ 份分别存储到 $X$ 块磁盘中，以实现快速的数据读取。
对于连续读写性能，RAID $0$ 可以提升速度到 $X$ 倍(相对于一块磁盘，但是其上限为南桥芯片组的带宽)，是所有 RAID 中速度最快的；
但是相对的，RAID $0$ 完全没有数据容错能力，一旦某个磁盘的数据损坏，则整个 RAID $0$ 都无法正常读取。</p></li>

<li><p style="text-align:justify; text-justify:inter-ideograph;">RAID $1$：也被称为镜像阵列。假设有 $X$ 块磁盘，RAID $1$ 通过将每一份数据都存储到每块磁盘中，相当于使用 $X-1$ 块磁盘进行备份。
它的优劣性和 RAID $1$ 刚好相反：拥有良好的容错性能，但是连续读写性能较低。</p></li>

<li><p style="text-align:justify; text-justify:inter-ideograph;">RAID $2$：使用<b>海明校验码</b>来实现数据校验(甚至恢复)。假设有 $X$ 块磁盘，则它将需要存储的数据平均分成 $X - \llcorner log_2X \lrcorner - 1$ 份分别存储到 $X$ 块磁盘中，
而剩下的 $\llcorner log_2X \lrcorner + 1$ 分别存储海明校验码的校验文件(存储在 $2^0, 2^1, ..., 2^{\llcorner log_2X \lrcorner}$)。
虽然提高了数据容错能力，但是读取和写入时都需要计算校验码，造成读写性能较低。</p></li>

<li><p style="text-align:justify; text-justify:inter-ideograph;">RAID $3$：使用<b>恢复码</b>来实现数据恢复。假设有 $X$ 块磁盘，RAID $3$ 通过将需要存储的数据平均分成 $X - 1$ 份分别存储到 $X - 1$ 块磁盘中，
而在最后一个磁盘存储数据的恢复码，当有一个磁盘损坏时，可以提高其余 $X - 2$ 个磁盘的数据 $+$ 恢复码实现数据恢复。
虽然提供了一块磁盘的数据容错能力，但是写入时需要计算恢复码，导致存储恢复码的磁盘写入性能较低，从而拖累了整体的性能。</p></li>

<li><p style="text-align:justify; text-justify:inter-ideograph;">RAID $4$：类似于 RAID $3$，同样使用<b>恢复码</b>来实现数据恢复；但是与其不同的是，RAID $3$ 是对每个数据计算恢复码，而 RAID $4$ 是对多份数据的总体计算恢复码。
举个例子，假设有 $X$ 块磁盘，有 $D_{1:M}$ $M$ 份数据，RAID $3$ 是将每个数据 $D_i$ 平均分成 $X - 1$ 份平均存储到 $X - 1$ 块磁盘中(第 $j$ 块磁盘存储 $D_i^j$)，
而在最后一个磁盘存储数据的恢复码 $R_i$，因此需要生成 $M$ 个恢复码；而RAID $3$ 是将每个数据 $D_i$ 平均分成 $X - 1$ 份平均存储到 $X - 1$ 块磁盘中(第 $j$ 块磁盘存储 $D_i^j$)，
然后将每个磁盘中的数据进行重新组合($D^j = [D_1^j,...,D_M^j]$)，形成一份数据块，然后在最后一个磁盘存储数据块 $[D^1,...,D^M]$ 的恢复码 $R$，因此只需要生成 $1$ 个恢复码。
因此 RAID $4$ 只需要计算较少的恢复码，提高了写入性能，但是在读取时，需要读取整个数据块(可能只需要其中的一部分)，导致读取性能较低。</p></li>

<li><p style="text-align:justify; text-justify:inter-ideograph;">RAID $5$：由于 RAID $3$ 中的恢复码磁盘会拖累整体性能，因此可以通过<b>平摊</b>的方式来减轻影响，即每次存储数据时，都随机地选择 $1$ 个磁盘作为恢复码磁盘。
这样可以避免刚好选到一个性能较差的磁盘一直作为恢复码磁盘，从而提高了读写性能。(它也是私有 NAS 服务器最常见的磁盘阵列形式)</p></li>

<li><p style="text-align:justify; text-justify:inter-ideograph;">RAID $6$：由于 RAID $5$ 只能容错一块磁盘，因此 RAID $6$ 使用 $2$ 块磁盘作为恢复码磁盘，同时在每次存储数据时，也是随机地选择 $2$ 个磁盘作为恢复码磁盘。</p></li></ol>

<h1>内存</h1>

<p style="text-align:justify; text-justify:inter-ideograph;"><b>多通道内存</b>：内存带宽 $=$ 内存核心频率 $\times$ 内存总线位宽 $\times$ 倍增系数。CPU 中与内存交互的部分称为 Memory Controller I/O (IMC)。
内存通道数由 IMC 决定。一般的 CPU 都只支持 $2$ 通道。此时，如果主板上有 $4$ 条内存槽，则 $1$ 和 $2$ 条内存槽为一通道，$3$ 和 $4$ 为二通道。
同时，一般建议使用对称的双通道内存配置(即一通道的内存容量为 $X$，则二通道的内存容量也应为 $X$)；如果使用非对称双通道内存配置(假设一通道的内存容量为 $X$，则二通道的内存容量也应为 $Y$，且 $X < Y$)，
此时一通道的所有内存容量 $X$ 和二通道的对应 $X$ 容量组成双通道，而二通道剩余的 $Y - X$ 容量自行组成单通道。</p>

<h1>其他硬件设备</h1>

<p style="text-align:justify; text-justify:inter-ideograph;"><b>视频采集卡</b>：在<b>硬件</b>层面进行视频采集。</p>