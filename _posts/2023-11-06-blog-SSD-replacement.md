---
title: '笔记本电脑固态硬盘更换扩容'
date: 23-11-06
permalink: /posts/2023/11/blog-ssd-replacement/
tags:
  - 电脑硬件
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要讲解只有一个固态硬盘(SSD)槽的笔记本电脑在需要扩容的简便操作(无需重装系统)。</p>

Problem
===

<p style="text-align:justify; text-justify:inter-ideograph;">在笔记本仅有一个 M2/SATA SSD 槽时，扩容的方法就是更换一个容量更大的 SSD。
这时候就涉及到如何将原有的 SSD 数据及 Windows 系统全部迁移到新的 SSD 上。
网上的教程大多只涉及到系统的迁移，即 Windows 系统所在盘(C 盘)的迁移，但是对于其他盘的迁移却没有涉及。
这篇博客主要基于 <a href="https://www.bilibili.com/video/BV1wu411L7eL/?spm_id_from=333.1007.top_right_bar_window_history.content.click" target="_blank">笔记本无损扩容教程</a>，
并将其扩展到整个 SSD 的迁移。(ps:：建议先看完上述的基础视频，视频 up 主真的非常良心)</p>

<p style="text-align:justify; text-justify:inter-ideograph;">首先需要准备 1 个 U 盘或移动硬盘，用于微 PE 工具箱安装；1 个新的 SSD 卡，需要将数据迁移到上面；
1 个移动硬盘盒用于 SSD 之间的数据传输(注意，移动硬盘盒和移动硬盘不是一个东西)。然后执行下面步骤：</p>

<p style="text-align:justify; text-justify:inter-ideograph;">第一步：检查电脑的 <b>BitLocker</b> 是否为关闭状态，若没有，则需要关闭，以便后续的数据迁移。
具体操作为打开电脑设置(快捷键 win 键 + X 键，然后再按 N 键)，搜索 Bit，选择<b>管理 BitLocker</b>，关闭每个盘(如 C, D 等)的 BitLocker。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">第二步：制作 PE 系统，在 <a href="https://www.wepe.com.cn/" target="_blank">微 PE 工具箱官网</a>下载<b>微 PE 工具箱</b>(作者选择 v2.3 版本)，
点开下好的 .exe 文件，插入准备好的 U 盘或移动硬盘，选择右下角<b>安装 PE 到 U 盘</b>，并在<b>待写入 U 盘</b>中选择插入的 U 盘或移动硬盘，选择立即安装 PE 到 U 盘，
等待安装完成即可。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">第三步：进入 PE 系统，将制作好的 PE 系统盘插入电脑(制作完成后不用拔出就好了)，打开电脑设置，搜索高级启动，选择<b>更改高级启动选项</b>，点击<b>立即重新启动</b>。
稍等片刻，电脑会变成蓝色选择界面，点击<b>使用设备</b> $\rightarrow$ <b>EFI USB Device</b>，电脑便进入PE 系统盘的内置 PE 系统。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">第四步：迁移数据，这一步请确保<b>电脑连接电源线</b>，不会半途没电。将新的 SSD 卡装入移动硬盘盒中，并插入系统(通常是使用 type-C 接口插入，这样传输速度快)。
注意，这里建议先进入 PE 系统盘后再插入移动硬盘盒，不然在使用 EFI USB Device 启动时，电脑会检测到 $2$ 个 USB 设备，会要求我们进行选择。
然后点击 PE 系统中的<b>分区助手</b>，点击左上角的<b>迁移系统到固态硬盘</b>，然后选择插入的新 SSD 
(如果你使用的 U 盘作为 PE 系统盘，则一般新 SSD 显示为硬盘 2，如果你使用的移动硬盘作为 PE 系统盘，则一般新 SSD 显示为硬盘 3；选中后会变蓝色)。
点击下一步后会看到一个<b>大小与位置</b>的显示区域，中间的白色进度条代表新的 SSD 的总容量，而绿色的进度条代表原 SSD 中的系统盘(即 C 盘)的已占用容量。
<b>注意</b>，这里<b>不是</b>把绿色圆圈拖到最后面，这表示将整个新的 SSD 全部作为系统盘(C 盘)，这会使得后面迁移非系统盘(如 D 盘等)时变得很繁琐。
这里需要将绿色圆圈拖到合适的位置(即你想给 C 盘分配的空间，假设为 $G_1$，需要<b>大于</b>原 SSD 的系统盘中已经使用的部分)，然后点击下一步并完成。
此时，我们的要求会显示在左上角的<b>提交</b>上，它并不会马上执行，而是等我们需要的一系列操作设置完后，再点击<b>提交</b>按钮才会执行。
但是你会看到执行后的效果，在新的 SSD 卡处(硬盘2/3)会显示将 $G_1$ 的空间分配给系统盘，而剩下的则是未分配空间。
然后，我们选中其他的非系统盘(每次选择一个)，然后在左下角的工具栏中选择<b>克隆分区</b>，选择<b>快速克隆分区</b>，该方式可以只克隆该分区中已使用部分的数据，
在弹出的目标克隆空间框内选择新的 SSD 卡处(硬盘2/3)刚刚剩下的未分配空间，
同样分配合适的空间，然后点击下一步完成。循环克隆非系统盘的步骤，直到原 SSD 中的所有盘都被克隆
(注意，如果当前克隆的非系统盘不是最后一个盘，依旧不能把所有空间都分配，还是需要留下足够的未分配空间给后续的非系统盘进行克隆，
但是每次克隆时分配的空间都要<b>大于</b>该盘已经使用的部分)。
最后，如果还剩下有未分配的空间，则可以点击该未分配的空间，再选择左下角的工具栏中的<b>创建分区</b>，将该未分配的空间单独创建成一个新盘。
执行完第四步的所有操作后，你会看到左上角<b>提交</b>按钮上的所有刚刚设置的操作，然后点击<b>提交</b>，选择<b>执行</b>，分区助手便会逐条执行我们的操作。
这时只需静静等待完成即可(迁移和克隆的时间通常较长，取决于文件大小和你的移动硬盘盒的传输速率)，完成后关闭分区助手并关机。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">如果你不小心跟随视频的做法将新的 SSD 的所有空间都设置为系统盘(通常显示为 E 盘)，可以选择该系统盘(即新的 SSD 的 E 盘)，选择左下角工具栏的<b>更改分区</b>，
将系统盘调成合适的大小，剩下的则会形成一个新的空分区。
然后选择该新的空分区，选择左下角工具栏的<b>删除分区</b>，将其重新设置为未分配，接着便可执行第四步中的克隆非系统盘的步骤(这就相比第四步多了更改分区和删除分区两个步骤，消耗时间也更久)。
或者重新执行第四步，将原先的系统盘重新覆盖。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">最后，只需要使用移动硬盘盒内的新的 SSD 替换掉原 SSD 即可完成 SSD 数据迁移的全部过程(注意，在拆卸和安装 SSD 时尽量拆掉电池，并保持手指干燥，避免静电)。</p>