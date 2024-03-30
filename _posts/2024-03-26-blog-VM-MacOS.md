---
title: 'VMware Workstation Pro 安装 MacOS 虚拟机'
date: 24-03-26
update: 24-03-30
permalink: /posts/2024/03/blog-vm-macos/
star: superior
tags:
  - 电脑软件
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要讲解如何在 VMware Workstation Pro 安装 MacOS 虚拟机。</p>

Problem
===

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要基于 <a href="https://blog.csdn.net/qq_45025572/article/details/108689543" target="_blank">window10安装Mac虚拟机详细教程</a>，
并将其扩展到最新的 MacOS 镜像安装</p>

<p style="text-align:justify; text-justify:inter-ideograph;">1. 安装 VMware Workstation Pro：解压 VMware-workstation-full-15.5.2.rar，点击 VMware-workstation-full-15.5.2-15785246.exe 进行安装（注意记住安装路径），然后解压 KeyGen-OnLyOnE.rar，运行 KeyGen.exe 获得许可验证码，运行 VMware Workstation Pro，输入许可验证码；</p>

<p style="text-align:justify; text-justify:inter-ideograph;">2. 解压 unlocker-3.0.8.rar，右击 win-install.cmd，选择以管理员身份运行（在运行期间可能需要点击任意键继续）；</p>

<p style="text-align:justify; text-justify:inter-ideograph;">3. （这时候先不要运行 VMware Workstation Pro）把 unlocker-3.0.8/iso 目录下的 darwin.iso 和 darwinPre15.iso 拷贝到 VMwarean Workstation Pro 的安装目录（正常是在 xxx/VMware\VMware Workstation/ 下），选择替换这两个文件；</p>

<p style="text-align:justify; text-justify:inter-ideograph;">4. 解压 MacOS 镜像，需要注意镜像是分块压缩的，因此需要将三个镜像压缩文件放在同一个文件夹下，然后使用 7-Zip/Winrar 等压缩软件解压其中某一个，就可以获得完整的镜像文件；</p>

<p style="text-align:justify; text-justify:inter-ideograph;">5. 接下来可以参考 <a href="https://blog.csdn.net/qq_45025572/article/details/108689543" target="_blank">window10安装Mac虚拟机详细教程</a>；</p>

<p style="text-align:justify; text-justify:inter-ideograph;">注意：出现 VMware Workstation 与 Device/Credential Guard 不兼容。在禁用 Device/Credential Guard 的问题，可以参考 <a href="https://blog.csdn.net/qq_37567470/article/details/129397491" target="_blank">这篇博客</a> 进行设置（记得先保存手头工作，因为它需要重启电脑）</p>

<p style="text-align:justify; text-justify:inter-ideograph;">注意：遇到鼠标键盘无法使用的问题，可以参考 <a href="https://blog.csdn.net/zhoupian/article/details/122659135" target="_blank">这篇博客</a> 进行设置（来自评论区的亲测有效，我没有遇到这种情况）</p>
