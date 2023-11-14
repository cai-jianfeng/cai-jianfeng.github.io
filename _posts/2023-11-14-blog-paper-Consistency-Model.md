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
更加详细的介绍推理可以参考 <a href="https://cai-jianfeng.github.io/posts/2023/11/blog-diffusion-model/" target="_blank">The Basic Knowledge of Diffusion Model (DM)</a>。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">Diffusion Model (DM continuous)：</p>

Method
===

![Consistency Model](/images/paper_Consistency_Model.png)

<p style="text-align:justify; text-justify:inter-ideograph;">目前 DM 模型最大的不足还是推理的速度太慢，由于需要进行多次逆扩散过程的迭代(即 multi-inference)，导致其推理时间较长。
本文便想构造方法实现一次 inference 直接生成图像。
最直观的方式是设计一个模型(如 U-net) $M_\theta$，输入给定的初始化噪声图像 $x_T$，输出预测的原始图像 $\hat{x}_0$，并使用 ground-truth $x_0$ 作为监督信号，
利用 MSE 函数计算损失 $L_{\theta} = \boldsymbol{E}[||\hat{x}_0 - x_0||_2^2]$ 进行训练。
当然，也可以和 <a href="https://cai-jianfeng.github.io/posts/2023/11/blog-diffusion-model/" target="_blank">The Basic Knowledge of Diffusion Model (DM)</a> 中提到的一样，
预测出 $x_T$ 所加的噪声 $\bar{\varepsilon_T}$ 来获得原始图像 $\hat{x}_0 = \dfrac{1}{\sqrt{\bar{\alpha}_T}}(x_T - \sqrt{1 - \bar{\alpha}_T}\bar{\varepsilon}_T)$。
但是这 $2$ 种方法的最大缺陷是性能不足，生成的图像的逼真度不够。
其中很大一部分原因是因为在没有其他条件的帮助下，直接让模型建模 $\mathcal{p}(x_0|x_T)/\mathcal{p}(\varepsilon_T|x_T)$ 的难度太大。
为此本文提出通过使用 probability flow (PF) ordinary differential equation (ODE) (概率流常微分方程) 的解轨迹(solution trajectory)来帮助模型学习。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">如上图(Figure 1)，DM (continuous) 模型的数学过程是一个 stochastic differential equation (SDE，随机微分方程)：</p>

<center>$dx_t = \mu(x_t,t)dt + \sigma(t) dw_t$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;">其中，假设 $x_t$ 的概率分布为 $\mathcal{p}_t(x) \Rightarrow p_0(x) = p_{data}(x), p_T(x) = \pi(x)$，
$\{w_t\}_{t \in [0,T]}$ 是 standard Brownain motion (标准布朗运动)。
SDE 有一个重要的性质：它存在一个 ODE 形式的方程，称为 PF ODE。
在本文的 DM 模型中，其 SDE 所对应 PF ODE 方程如下：</p>

<center>$dx_t = [\mu(x_t,t)dt - \dfrac{1}{2} \sigma(t)^2 \triangledown log\mathcal{p}_t(x_t)]dt$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;">其中，$\triangledown log\mathcal{p}_t(x_t)$ 是 $\mathcal{p}_t(x_t)$ 的 score function。
为了方便建模，本文将方程进行简化：令 $\mu(x_t,t) = 0,\ \sigma(t) = \sqrt{2t},\ p_t(x) = p_{data}(x) \otimes \mathcal{N}(0, T^2\boldsymbol{I})$。
为了实现对 PF ODE 解轨迹的求解，本文首先通过 score match 训练一个 score model $s_{\phi}(x_t,t) \approx \triangledown logp_t(x_t)$。
然后将其代入方程，将方程简化为常系数微分方程(称为 empirical PF ODE)：</p>

<center>$\dfrac{dx}{dt} = -ts_{\phi}(x_t,t)$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;">然后，采样 $\hat{x}_T \sim \pi = \mathcal{N}(0,T^2\boldsymbol{I})$ 初始化方程，
然后使用 numerical ODE solver (数值常微分方程求解器，例如 Euler/Heun solver)求解方程，从而获得整个解轨迹 $\{\hat{x}_t\}_{t \in [0,T]}$，
其中 $\hat{x}_0$ 可以近似为在数据分布 $\mathcal{p}_{data}(x)$ 中的采样。
为了数值稳定性，通常当 $t = \epsilon, \epsilon \in R^+\ and\ \epsilon \rightarrow 0$ 时就停止计算，并将最终的 $\hat{x}_\epsilon$ 作为 $\hat{x}_0$ 的近似，
则整个解轨迹就变成 $\{\hat{x}_t\}_{t \in [\epsilon,T]}$。本文中使用 $T = 80, \epsilon = 0.002$。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">有了解轨迹，本文便提出了 <b>consistency model</b> 来利用它进行一步 inference 的学习。
具体而言，定义 consistency model 为 $\boldsymbol{f_\theta}:(x_t,t) \mapsto x_\epsilon$。它具有 self-consistency (自一致)的性质：
即 $\boldsymbol{f_\theta}(x_t,t) = \boldsymbol{f_\theta}(x_{t'},t'), \forall t,t' \in [\epsilon, T]$ 
(对于任意在相同 PF ODE 解轨迹上的输入对 $(x_t,t)$，其输出一致，都是 $x_\epsilon$)。
这就使得模型具有一定的限制。其中最主要的限制便是 boundary condition：$\boldsymbol{f_\theta}(x_\epsilon,\epsilon) = x_\epsilon$，即 $\boldsymbol{f_\theta}(·,\epsilon)$ 是一个 identity function (恒等函数)。
为了尽可能减少它对模型的限制(包括输入输出，结构等)，本文提出 $2$ 种方法来构建 consistency model，称为 <b>parameterization (参数化)</b>：</p>

$$\boldsymbol{f_\theta}(x,t) = \begin{cases} x, & t = \epsilon \\ F_\theta(x,t), & t \in (\epsilon, T]\end{cases}$$

$$\boldsymbol{f_\theta}(x,t) = c_{skip}(t)x + c_{out}(t)F_\theta(x,t)$$

<p style="text-align:justify; text-justify:inter-ideograph;">其中，$F_\theta(x,t)$ 是 free-form 的深度神经网络；
第 $2$ 种方法的 $c_{skip}(t)$ 和 $c_{out}(t)$ 都是可微函数(differentiable)，且满足 $c_{skip}(\epsilon)=1, c_{out}(\epsilon)=0$。
由于第 $2$ 种方法和 DM 模型相似(DM 模型是将噪声 $\varepsilon$ 分离出来实现参数化，即 $f_\theta(x) = x + \varepsilon_\theta(x)$)，
许多流行的 DM 模型架构都可以直接使用，因此本文选择第 $2$ 种方法参数化模型。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">而在训练完成后，模型可以进行<b>一步 inference </b>实现图像生成。
具体而言，给定一个训练好的 consistency model $\boldsymbol{f_\theta}(·,·)$，从高斯分布空间中随机采样一个噪声 $\hat{x}_T \sim \mathcal{N}(0,T^2\boldsymbol{I})$，
然后通过模型生成最终的图像 $\hat{x}_\epsilon = \boldsymbol{f_\theta}(\hat{x}_T,T)$。
更重要的是，模型也可以进行<b>多步 inference</b> 提高图像的生成质量。它通过多次交替去噪(denoise)和噪声注入(noise)步骤来使用 consistence model 精细化图像。具体算法如下图(Algorithm 1)。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">接下来便是模型架构和训练问题。对于模型架构，如前述，$F_\theta(x,t)$ 可以使用主流的 DM 模型(如 U-net)；
而对于 $c_{skip}(t)$ 和 $c_{out}(t)$，满足 $c_{skip}(\epsilon)=1, c_{out}(\epsilon)=0$ 的函数也有很多，本文遵循 EDM 的方式，使用：</p>

<center>$c_{skip}(t) = \dfrac{\sigma_{data}^2}{(t - \epsilon)^2 + \sigma_{data}^2}, c_{out}(t) = \dfrac{\sigma_{data}(t - \epsilon)}{\sqrt{\sigma_{data}^2 + t^2}}, \sigma_{data} = 0.5$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;">而对于训练问题，本文提出了 $2$ 种训练方法。第一种是 <b>Distillation</b>，它通过 distill 一个预训练好的 score model $s_\Theta(x,t)$ 来训练 consistency model。
具体而言，首先将区间(time horizon) $[\epsilon, T]$ 划分为 $N - 1$ 个子区间：</p>

<center>$[t_1,t_2],...,[t_{N-1},t_N], t_1 = \epsilon < t_2 <...<t_N = T$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<center>$t_i = (\epsilon^{\dfrac{1}{\rho}} + \dfrac{i-1}{N-1}(T^{\dfrac{1}{\rho}} - \epsilon^{\dfrac{1}{\rho}}))^{\rho}, \rho = 7$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;">然后使用 numerical ODE solver 求解 empirical PF ODE，得到 $\hat{x}^{\mathcal{\Phi}}_{t_n}$：</p>

<center>$\hat{x}^{\mathcal{\Phi}}_{t_n}:=x_{t_{n+1}} + (t_n - t_{n+1} \Phi(x_{t_{n+1}}, t_{n+1}; \mathcal{\Phi}))$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">其中 $\Phi(...; \mathcal{\Phi}))$ 表示 ODE solver 的更新函数(update function)。例如，如果使用 Euler solver，$\Phi(x,t; \mathcal{\Phi})) = -ts_\mathcal{\Phi}(x,t)$。
同时，由于前述 SDE 和 PF ODE 的关联性，可以通过先采样 $x \sim \mathcal{p}_data$，然后向 $x$ 添加高斯噪声，接着沿着 ODE 轨迹的分布进行采样。
具体而言，给定 $x$，首先使用 SDE 的过渡密度分布 $\mathcal{N}(x,t^2_{n+1}\boldsymbol{I})$ 采样得到 $x_{t_{n+1}}$；然后使用 numerical ODE solver 计算 $\hat{x}^{\mathcal{\Phi}}_{t_n}$，
与 $x_{t_{n+1}}$ 组成 adjacent data points $(\hat{x}^{\mathcal{\Phi}}_{t_n}, x_{t_{n+1}})$。
最后，通过最小化数据对 $(\hat{x}^{\mathcal{\Phi}}_{t_n}, x_{t_{n+1}})$ 的输出差异性(consistency distillation loss)来训练 consistency model：</p>

$$L_{CD}^N(\boldsymbol{\theta},\boldsymbol{\theta}^-;\mathcal{\Phi}):=E[\lambda(t_n)d(\boldsymbol{f_\theta}(x_{t_{n+1}},t_{n+1}), \boldsymbol{f_{\theta^-}}(\hat{x}^{\mathcal{\Phi}}_{t_n}, t_n))]$$

<center>$x \sim p_{data}, n \sim \mathcal{U}[1,N-1],x \sim \mathcal{N}(x;t^2_{n+1}\boldsymbol{I})$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;">其中，$\lambda(·) \in R^+$ 表示正值权重函数；$\boldsymbol{\theta}^-$ 表示 $\boldsymbol{\theta}$ 的 EMA (指数移动平均)：
$\boldsymbol{\theta}^- \leftarrow stopgrad(\mu\boldsymbol{\theta}^- + (1-\mu)\boldsymbol{\theta})$；
$d(·,·)$ 表示度量函数，满足 $\forall x,y: d(x,y) \geq 0$ and $d(x,y)=0$ if and only if $x=y$。
本文使用 $\lambda(t_n) \equiv 1$，$d(x,y) = \mathcal{l}_2:||x-y||_2^2/\mathcal{l}_1:||x-y||_1/LPIPS$。具体算法如下图(Algorithm 2)。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">第二种是 <b>Isolation</b>，即不需要任何额外的预训练模型，从头开始训练。回顾上述的 Distillation，
它将预训练好的 score model  近似为 $\triangledown log\mathcal{p}_t(x_t)$。
只需要找到不依赖 $s_{\mathcal{\Phi}}(x,t)$ 的 $\triangledown log\mathcal{p}_t(x_t)$ 函数即可实现 Isolation 训练。
为此，本文找到了一个 unbiased estimator $\triangledown log\mathcal{p}_t(x_t) = -E[\dfrac{x_t-x}{t^2}|x_t]$。
然后便可使用相似的 consistency training loss 来训练 consistency model：</p>

$$L_{CD}^N(\boldsymbol{\theta},\boldsymbol{\theta}^-;\mathcal{\Phi})=L_{CT}^N(\boldsymbol{\theta},\boldsymbol{\theta}^-)=E[\lambda(t_n)d(\boldsymbol{f_\theta}(x + t_{n+1}z,t_{n+1}), \boldsymbol{f_\theta^-}(x + t_nz,t_{n}))]$$

<p style="text-align:justify; text-justify:inter-ideograph;">其中，$z \sim \mathcal{N}(0,\boldsymbol{I})$。具体算法如下图(Algorithm 3)。</p>

![Comsistency Model Algorithm](/images/paper_Consistency_Model_Algorithm.png)