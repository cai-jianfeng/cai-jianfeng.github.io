---
title: 'Consistency Model'
date: 23-11-14
permalink: /posts/2023/11/blog-paper-consistency-model/
tags:
  - 论文阅读
---

<p style="text-align:justify; text-justify:inter-ideograph;"> 论文题目：<a href="https://openreview.net/forum?id=FmqFfMTNnv" target="_blank" title="Consistency Model">Consistency Models</a></p>

<p style="text-align:justify; text-justify:inter-ideograph;">发表会议：International Conference on Machine Learning (ICML 2023)</p>

第一作者：Yang Song (OpenAI)

Question
===

<p style="text-align:justify; text-justify:inter-ideograph;">如何提高图像生成模型的生成速度，同时尽可能保持高性能(逼真)</p>

Preliminary
===

<p style="text-align:justify; text-justify:inter-ideograph;">Diffusion Model (DM dispersed)：如下图(Figure 1)，扩散模型是近年来最热的图像生成思想。
将任意一张图像 $I_0$ 进行噪声添加，每次添加一个服从 $N(0,1)$ 分布的随机噪声 $\epsilon_t$，获得含有噪声的图像 $I_t$，则进行了无数次后，原本的图像就会变成一个各向同性的随机高斯噪声。
按照这个理论，我们对一个各向同性的随机高斯噪声每次添加一个特定的服从 $N(0,1)$ 分布的噪声 $\epsilon_t'$，获得噪声量减少的图像 $I_t'$，则经过足够多次后便可获得一张逼真的图像 $I_0'$。
为此，我们可以选择任意一个图像生成模型 $M_G$ (例如 U-net 等)，第 t 次时输入第 t-1 次生成的图像 $I_{t-1}'$，输出生成的图像 $I_t'$。
由于每次使用的是同一个图像生成模型 $M_G$，所以需要输入时间步信息 $t$ 来告诉模型现在是生成的第几步
(一个直观的理解是在随机噪声恢复成原始图像的前期，模型更关注图像的整体恢复，而后期则更加关注图像的细节恢复，所以需要时间步信息告诉模型现在是偏前期还是后期)。
同时通过 $I_{t-1}$ 直接生成 $I_t$ 较为困难，因为它需要生成正确其每个像素值，而每次添加的噪声 $\epsilon_t$ 较容易预测。
因此可以更换为每次输入 $I_{t-1}$，模型预测第 $t$ 次所加的噪声 $\epsilon_t$。所以损失函数为 $L = E_{z_0, t, c_t, \epsilon \sim N(0,1)}[||\epsilon - \epsilon_{\theta}(z_t, t, c_t)||_2^2]$。
其中 $z_0$ 是原始噪声(即一开始输入的图像)，而 $z_t$ 是第 $t$ 步输出的图像, $\epsilon_{\theta}(z_t, t, c_t)$ 为模型预测的第 $t$ 步所加的噪声。
更加详细的介绍推理可以参考 <a href="https://cai-jianfeng.github.io/posts/2023/11/blog-diffusion-model/">The Basic Knowledge of Diffusion Model (DM)</a>。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">Diffusion Model (DM continuous)：</p>

Method
===

![Consistency Model](/images/paper_Consistency_Model.png)

<p style="text-align:justify; text-justify:inter-ideograph;">目前 DM 模型最大的不足还是推理的速度太慢，由于需要进行多次逆扩散过程的迭代(即 multi-inference)，导致其推理时间较长。
本文便想构造方法实现一次 inference 直接生成图像。
最直观的方式是设计一个模型(如 U-net) $M_\theta$，输入给定的初始化噪声图像 $x_T$，输出预测的原始图像 $\hat{x}_0$，并使用 ground-truth $x_0$ 作为监督信号，
利用 MSE 函数计算损失 $L_{\theta} = \mathbf{E}[||\hat{x}_0 - x_0||_2^2]$ 进行训练。
当然，也可以和 <a href="https://cai-jianfeng.github.io/posts/2023/11/blog-diffusion-model/">The Basic Knowledge of Diffusion Model (DM)</a> 中提到的一样，
预测出 $x_T$ 所加的噪声 $\varepsilon_T$ 来获得原始图像 $\hat{x}_0 = \dfrac{1}{\sqrt{\bar{\alpha}_T}}(x_T - \sqrt{1 - \bar{\alpha}_T}\bar{\varepsilon}_T)$。
但是这 $2$ 种方法的最大缺陷是性能不足，生成的图像的逼真度不够。
其中很大一部分原因是因为在没有其他条件的帮助下，直接让模型建模 $\mathcal{p}(x_0|x_T)/\mathcal{p}(\varepsilon_T|x_T)$ 的难度太大。
为此本文提出通过使用 probability flow (PF) ordinary differential equation (ODE) (概率流常微分方程) 的解轨迹(solution trajectory)来帮助模型学习。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">如上图(Figure 1)，DM (continuous) 模型的数学过程是一个 stochastic differential equation (SDE，随机微分方程)：</p>

<center>$dx_t = \mu(x_t,t)dt + \sigma(t) dw_t$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;">其中，$\{w_t\}_{t \in [0,T]}$ 是 standard Brownain motion (标准布朗运动)。
假设 $x_t$ 的概率分布为 $\mathcal{p}_t(x) \Rightarrow p_0(x) = p_{data}(x), p_T(x) = \pi(x)$，SDE 有一个重要的性质：它存在一个 ODE 称为 PF ODE。
在本文的 DM 模型中，其 SDE 所对应 PF ODE 解轨迹如下：</p>

<center>$dx_t = [\mu(x_t,t)dt - \dfrac{1}{2} \sigma(t)^2 \triangledown log\mathcal{p}_t(x_t)]dt$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;">其中，$\triangledown log\mathcal{p}_t(x_t)$ 是 $\mathcal{p}_t(x_t)$ 的 score function。
本文将方程进行简化：$\mathcal{p}_t(x_t) = 0, \mathcal{p}_t(x_t) = \sqrt{2t}, p_t(x) = p_{data}(x) \otimes \mathcal{N}(0, T^2\mathbf{I}$。
为了实现对 PF ODE 解轨迹的求解，本文首先通过 score match 训练一个 score model $s_{\Phi}(x_t,t) \approx \triangledown logp_t(x_t)$。
然后将其代入方程，将方程简化为常系数微分方程(称为 empirical PF ODE)：</p>

<center>$\dfrac{dx}{dt} = -ts_{\Phi}(x_t,t)$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;">然后，采样 $\hat{x}_T \sim \pi = \mathcal{N}(0,T^2\mathbf{I})$ 初始化方程，
然后使用 numerical ODE solver (数值常微分方程求解器，例如 Euler/Heun solver)求解方程，从而获得整个解轨迹 $\{\hat{x}_t\}_{t \in [0,T]}$，
其中 $\hat{x}_0$ 可以近似为在数据分布 $\mathcal{p}_{data}(x)$ 中的采样。
为了数值稳定性，通常当 $t = \epsilon, \epsilon \in R^+\ and\ \epsilon \rightarrow 0$时就停止计算，并将最终的 $\hat{x}_\epsilon$ 作为 $\hat{x}_0$ 的近似，
则整个解轨迹就变成 $\{\hat{x}_t\}_{t \in [\epsilon,T]}$。本文中使用 $T = 80, \epsilon = 0.002$。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">有了解轨迹，本文便提出了 <b>consistency model</b> 来利用它进行一步 inference 的学习。
具体而言，定义 consistency model 为 $\mathbf{f_\theta}:(x_t,t) \mapsto x_\epsilon$。它具有 self-consistency (自一致)的性质：
$\mathbf{f_\theta}(x_t,t) = \mathbf{f_\theta}(x_{t'},t'), \forall t,t' \in [\epsilon, T]$ 
(对于任意在相同 PF ODE 解轨迹上的输入对 $(x_t,t)$，其输出一致，都是 $x_\epsilon$)。
这就使得模型具有一定的限制。其中最主要的限制便是 boundary condition：$\mathbf{f_\theta}(x_\epsilon,\epsilon) = x_\epsilon$，即 $\mathbf{f_\theta}(·,\epsilon)$ 是一个 identity function (恒等函数)。
为了尽可能减少它对模型的限制(包括输入输出，结构等)，本文提出 $2$ 种方法来构建 consistency model，称为 <b>parameterization (参数化)</b>：</p>

$$\mathbf{f_\theta}(x,t) = \begin{cases} x, & t = \epsilon \\ F_\theta(x,t), & t \in (\epsilon, T]\end{cases}$$

$$\mathbf{f_\theta}(x,t) = c_{skip}(t)x + c_{out}(t)F_\theta(x,t)$$

<p style="text-align:justify; text-justify:inter-ideograph;">其中，$F_\theta(x,t)$ 是 free-form 的深度神经网络；
第 $2$ 种方法的 $c_{skip}(t)$ 和 $c_{out}(t)$ 都是可微函数(differentiable)，且满足 $c_{skip}(\epsilon)=1, c_{out}(\epsilon)=0$。
由于第 $2$ 种方法和 DM 模型相似(DM 模型是将噪声 $\varepsilon$ 分离出来实现参数化，即 $f_\theta(x) = x + \varepsilon_theta(x)$)，
许多流行的 DM 模型架构都可以直接使用，因此本文选择第 $2$ 种方法参数化模型。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">而在训练完成后，模型可以进行<b>一步 inference </b>实现图像生成。
具体而言，给定一个训练好的 consistency model $\mathbf{f_\theta}(·,·)$，从高斯分布空间中随机采样一个噪声 $\hat{x}_T \sim \mathcal{N}(0,T^2\boldsymbol{I})$，
然后通过模型生成最终的图像 $\hat{x}_\epsilon = \boldsymbol{f_\theta}(\hat{x}_T,T)$。</p>

![Comsistency Model Algorithm](/images/paper_Consistency_Model_Algorithm.png)