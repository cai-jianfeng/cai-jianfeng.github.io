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

<p style="text-align:justify; text-justify:inter-ideograph;">1. 安装 VMware Workstation Pro：解压 VMware-workstation-full-15.5.2.rar，
点击 VMware-workstation-full-15.5.2-15785246.exe 进行安装（注意<b>记住安装路径</b>），
然后解压 KeyGen-OnLyOnE.rar，运行 KeyGen.exe 获得许可验证码，在 VMware Workstation Pro 后，输入许可验证码；</p>

<p style="text-align:justify; text-justify:inter-ideograph;">2. 解压 unlocker-3.0.8.rar，右击 win-install.cmd，选择以<b>管理员身份</b>运行（在运行期间可能需要点击任意键退出）；</p>

<p style="text-align:justify; text-justify:inter-ideograph;">3. （这时候先不要运行 VMware Workstation Pro）
把 unlocker-3.0.8/iso 目录下的 darwin.iso 和 darwinPre15.iso 拷贝到 VMwarean Workstation Pro 的安装目录
（正常是在 [你选择的安装路径]\VMware\VMware Workstation\ 下），选择替换这两个文件；</p>

![file replace](/images/VM_MacOS_file_replace.png)

<p style="text-align:justify; text-justify:inter-ideograph;">4. 下载并解压 MacOS 镜像：对于 EI Capitan 10.11 Install 镜像，需要注意镜像是分块压缩的，
因此需要将三个镜像压缩文件(EI Capitan 10.11 Install part1/2/3.rar)全部下载，并放在同一个文件夹下，
然后使用 7-Zip/Winrar 等压缩软件解压其中某一个，就可以获得完整的镜像文件；
对于 CD Install macOS Big Sur 11.6.7(20G630) 镜像，直接下载 CD Install macOS Big Sur 11.6.7(20G630).iso 即可。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">5. 新建虚拟机（下面没有明确说明的设置均为直接下一步）：首先打开 VMware Workstation Pro，在界面点击创建新的虚拟机或者使用快捷键 Ctrl+N 进行创建：</p>

![new vm](/images/VM_MacOS_newvm.png)

<p style="text-align:justify; text-justify:inter-ideograph;">然后在新建虚拟机向导窗口中选择"自定义（高级）"并点击下一步：</p>

![new vm 2](/images/VM_MacOS_newvm_2.png)

![new vm 3](/images/VM_MacOS_newvm_3.png)

<p style="text-align:justify; text-justify:inter-ideograph;">在新建虚拟机向导中选择“<b>稍后安装操作系统</b>”并点击下一步：</p>

![new vm 4](/images/VM_MacOS_newvm_4.png)

<p style="text-align:justify; text-justify:inter-ideograph;">选择 Apple Mac OS X(M)，版本选择和你的 MacOS 镜像相对应的（EI Capitan 10.11 Install 镜像选择 OS X 10.11；
CD Install macOS Big Sur 11.6.7(20G630) 镜像选择 macOS 10.14），点击下一步：</p>

<p style="text-align:justify; text-justify:inter-ideograph;">（注意：1、如果这个页面没有 Apple Mac OS X(M)，说明虚拟机解锁不成功，即 2, 3 步骤不成功。
2、如果版本中没有 OS X 10.11/macOS 10.14 说明 VMware Workstation Pro 版本过低，请下载新版本）</p>

![new vm 5](/images/VM_MacOS_newvm_5.png)

<p style="text-align:justify; text-justify:inter-ideograph;">选择要存放虚拟机的位置（一般建议在 VMware Workstation Pro 安装目录的同级目录下），然后点击下一步：</p>

![new vm 6](/images/VM_MacOS_newvm_6.png)

<p style="text-align:justify; text-justify:inter-ideograph;">接下来全部点击下一步（设置处理器数量、分配内存、网络连接、I/O 控制器、磁盘类型、磁盘容量(最好选择 40 G 以上)、磁盘文件），最后点击完成就创建好了空白虚拟机：</p>

![new vm 7](/images/VM_MacOS_newvm_7.png)

![new vm 10](/images/VM_MacOS_newvm_10.png)

![new vm 13](/images/VM_MacOS_newvm_13.png)

<p style="text-align:justify; text-justify:inter-ideograph;">6. <b>配置虚拟机</b>：找到刚才创建的虚拟机并点击，然后点击右侧的“<b>编辑虚拟机设置</b>”：</p>

<p style="text-align:justify; text-justify:inter-ideograph;">如果不小心关闭了虚拟机的界面，可以点击 <b>打开虚拟机 - 找到创建虚拟机时的文件目录 - 点击打开 .vmx 文件</b> 来重新打开虚拟机界面：</p>

![configuration](/images/VM_MacOS_set.png)

![configuration 2](/images/VM_MacOS_configurate_2.png)

<p style="text-align:justify; text-justify:inter-ideograph;">点击 CD/DVD 选中右侧的“<b>使用 ISO 镜像文件(M)</b>”，点击浏览，选择步骤 4 中下载好的镜像
（EI Capitan 10.11 Install 镜像是 cdr 结尾的，需要把右下角调成所有文件（*.*））：</p>

![configuration 3](/images/VM_MacOS_configurate_3.png)

<p style="text-align:justify; text-justify:inter-ideograph;">接着点击底部的<b>添加</b>，选择<b> CD/DVD 驱动器</b>，点击<b>完成</b>，会出现新的 CD/DVD 2(SATA)：</p>

![configuration 4](/images/VM_MacOS_configurate_4.png)

<p style="text-align:justify; text-justify:inter-ideograph;">点击 CD/DVD 2(SATA)，选择“<b>使用iso镜像</b>”，点击浏览，选择 VMware Workstation Pro 安装目录下的 darwin.iso，最后点击确定：</p>

![configuration 5](/images/VM_MacOS_configurate_5.png)

<p style="text-align:justify; text-justify:inter-ideograph;">7. 修改虚拟机目录下的配置文件：找到创建虚拟机时的文件目录，使用<b>记事本</b>打开 .vmx 文件
（EI Capitan 10.11 Install 镜像是 OS X 10.11.vmx；CD Install macOS Big Sur 11.6.7(20G630) 镜像选择 macOS 10.14.vmx），
在 smc.present = "TRUE" 下一行添加一行代码 smc.version = 0（注意，没有引号，不然到时候启动虚拟机会直接蓝屏重启）：</p>

![configuration 6](/images/VM_MacOS_configurate_6.png)

<p style="text-align:justify; text-justify:inter-ideograph;">8. 配置 MacOS 系统（下面以 CD Install macOS Big Sur 11.6.7(20G630) 镜像为例，
EI Capitan 10.11 Install 镜像可以参考 <a href="https://blog.csdn.net/qq_45025572/article/details/108689543" target="_blank">window10安装Mac虚拟机详细教程</a>）：
关闭 VMware Workstation Pro，重新右击软件，选择用<b>管理员</b>身份打开 VMware Workstation Pro，打开虚拟机，点击“<b>开启此虚拟机</b>”：</p>

![OS configuration](/images/VM_MacOS_os_configurate.png)

<p style="text-align:justify; text-justify:inter-ideograph;">设置语言：选择简体中文，点击右下角箭头（下一步）：</p>

![OS configuration2](/images/VM_MacOS_os_configurate2.png)

<p style="text-align:justify; text-justify:inter-ideograph;">设置时间：点击“<b>实用工具</b>”，点击<b>终端</b>，查询安装的 Mac 系统发布时间，输入相应的时间
(CD Install macOS Big Sur 11.6.7(20G630) 镜像 可以输入 <b>date 070512052023.03</b>)，点击回车设置，点击左上角终端，选择退出终端：</p>

![OS configuration3](/images/VM_MacOS_os_configurate4.png)

<p style="text-align:justify; text-justify:inter-ideograph;">抹除磁盘：选择<b>磁盘工具</b>，选择第一个 Vmware Virtual SATA Hard Drive Media，选择<b>抹掉</b>，设置名称，
选择格式为 <b>Mac OS扩展(日志式)</b>，点击抹掉，抹掉后点击<b>完成</b>，最后点击左上角红色<b>关闭</b>按钮：</p>

![OS configuration6](/images/VM_MacOS_os_configurate6.png)

![OS configuration6](/images/VM_MacOS_os_configurate3.png)

![OS configuration7](/images/VM_MacOS_os_configurate7.png)

<p style="text-align:justify; text-justify:inter-ideograph;">安装 MacOS 系统：选择安装 macOS Big Sur，点击<b>两次继续</b>，点击<b>两次同意</b>，
选择 MacOS 磁盘（如果之前没有抹掉磁盘，这里的磁盘空间就会不足），点击继续：</p>

![OS configuration8](/images/VM_MacOS_os_configurate8.png)

![OS configuration9](/images/VM_MacOS_os_configurate11.png)

<p style="text-align:justify; text-justify:inter-ideograph;">macOS Big Sur 安装完成后，在<b>选择国家和地区</b>里划到最下面选择<b>中国大陆</b>，点击<b>四次继续</b>：</p>

![OS incontent](/images/VM_MacOS_incontent.png)

<p style="text-align:justify; text-justify:inter-ideograph;">在<b>迁移助理</b>里选择<b>以后</b>，在<b>Apples ID</b>里选择<b>稍后设置</b>，点击<b>跳过</b>，点击<b>两次同意</b>：</p>

![OS incontent2](/images/VM_MacOS_incontent2.png)

<p style="text-align:justify; text-justify:inter-ideograph;">在<b>迁移助理</b>里选择<b>以后</b>，在<b>Apples ID</b>里选择<b>稍后设置</b>，点击<b>跳过</b>，点击<b>两次同意</b>：</p>

<p style="text-align:justify; text-justify:inter-ideograph;">设置自己的<b>全名、账户名称</b>（一般账号名称和全名默认一致）和<b>密码</b>：</p>

![OS incontent3](/images/VM_MacOS_incontent4.png)

<p style="text-align:justify; text-justify:inter-ideograph;">快捷设置点击继续；分析里取消勾选，点击继续；屏幕使用时间点击稍后设置；Siri 里取消勾选，点击继续；选取自己喜欢的外观，点击继续，最后等待设置完成：</p>

![OS incontent4](/images/VM_MacOS_incontent5.png)

![OS incontent5](/images/VM_MacOS_incontent6.png)

<p style="text-align:justify; text-justify:inter-ideograph;">9. 关闭 Mac 虚拟机，点击“<b>编辑虚拟机设置</b>”，点击第一个 “CD/DVD（SATA）”将“<b>使用ISO镜像文件</b>”改为“<b>使用物理驱动器</b>”（因为系统已经安装在本地磁盘）：</p>

![Mac final](/images/VM_MacOS_change.png)

<p style="text-align:justify; text-justify:inter-ideograph;">注意：出现 "VMware Workstation 与 Device/Credential Guard 不兼容。在禁用 Device/Credential Guard" 的问题，可以参考 <a href="https://blog.csdn.net/qq_37567470/article/details/129397491" target="_blank">这篇博客</a> 进行设置（记得先保存手头工作，因为它需要重启电脑）</p>

<p style="text-align:justify; text-justify:inter-ideograph;">注意：遇到鼠标键盘无法使用的问题，可以参考 <a href="https://blog.csdn.net/zhoupian/article/details/122659135" target="_blank">这篇博客</a> 进行设置（来自评论区的亲测有效，我没有遇到这种情况）</p>
