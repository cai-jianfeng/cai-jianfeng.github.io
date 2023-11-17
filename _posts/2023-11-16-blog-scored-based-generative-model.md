---
title: 'The Basic Knowledge of Scored-based Generative Model'
date: 23-11-16
permalink: /posts/2023/11/blog-score-based-generative-model/
tags:
  - 深度学习基础知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客参考了<a href="https://yang-song.net/blog/2021/score/#connection-to-diffusion-models-and-others" target="_blank">
Generative Modeling by Estimating Gradients of the Data Distribution</a>，
详细讲述了最近大火的 Diffusion Model 的另一个理解/推理角度: Score-based Generative Model 的数学原理及编程。
(ps：建议先看完上述的 Generative Modeling by Estimating Gradients of the Data Distribution 博客，虽然是全英文的，但是写的十分详细，且简单易懂，真的非常良心)</p>

Score-Based Generative Model 的基本原理
===

<p style="text-align:justify; text-justify:inter-ideograph;">给定一个数据集 $D = \{x_1,...,x_N\}, x_i \sim p(x)$，
生成模型(generative model) $M_\theta(·)$ 的目标是要拟合 $p(x)$，使得可以通过采样 $M_\theta(·)$ 来生成新的数据 $\hat{x}$。
最简单直接的一个方式是训练一个模型 $M_\theta(x)$ 拟合给定数据集 $D$ 的概率分布函数(p.d.f.) $p(x)$，
即 $\boldsymbol{M_\theta(x) = p_\theta(x) = \dfrac{e^{-f_\theta(x)}}{Z_\theta} \approx p(x),Z_\theta > 0, \int p_\theta(x)dx = 1}$。
其中，$\boldsymbol{f_\theta(x)}$ 通常被称为 unnormalized probabilistic model/energy-based model。
然后使用常用的训练方式(maximize log-likehood (最大化对数似然))来训练模型 $M_\theta(x) = p_\theta(x)$：$L_\theta = \underset{\theta}{max}\sum_{i=1}^{N}{log\ \boldsymbol{p_\theta(x_i)}}$。
这样便可以训练一个生成模型来生成数据。但是这种做法的最大局限性在于 $p_\theta(x)$ 必须满足 $\int p_\theta(x)dx = 1$，
这就使得必须计算出 normalizing constant $Z_\theta = \int e^{-f_\theta(x)}dx$ 来限制 $p_\theta(x)$，
而对于任何一个 general $f_\theta(x)$ 来说这都是一个经典的难以计算的量。
因此常见的方法要么限制模型的架构使得 $Z_\theta$ 可以计算(限制了模型的灵活性)，要么使用其他方式近似 $Z_\theta$ (存在偏差)，它们都存在一定的问题。
但是如果我们能找到一个<b>替代函数</b>，使得函数中仅包含 $f_\theta(x)$，那么就可以避免计算 $Z_\theta$，这样就可以既保持模型架构的灵活性，又不引入偏差。
本文便找到了一个满足条件的简单函数 $\triangledown_xlog\ p(x)$ (称为 $p(x)$ 的 <b>score function</b>)可以作为替代函数。其证明推导如下：</p>

<center>$\triangledown_xlog\ p_\theta(x) = \triangledown_xlog \dfrac{e^{-f_\theta(x)}}{Z_\theta} = -\triangledown_xf_\theta(x) -\underset{=0}{\underbrace{ \triangledown_xlog\ Z_\theta}} = -\triangledown_xf_\theta(x)$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;">可以看到，$\triangledown_xlog\ p(x)$ 中没有包含 $Z_\theta$。那么我们只需要训练一个模型 $s_\theta(x)$ 拟合 $\triangledown_xlog\ p(x)$ 就可以了
(且该拟合模型没有任何限制条件，除了需要保证输入输出维度相同)，
即 $\boldsymbol{s_\theta(x) \approx \triangledown_xlog\ p(x)}$，其中的拟合模型 $s_\theta(x)$ 称为 score-based model。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">而在训练方面，由于 $\triangledown_xlog\ p(x)$ 仍然是一个密度分布(unnormalized)，则可以使用常见的 minimize <b>Fisher divergence</b> 来训练，即：</p>

<center>$L_\theta = \mathbb{E}_{p(x)}[||\triangledown_xlog\ p(x) - s_\theta(x)||_2^2]$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;">(注意，由于 $\triangledown_xlog\ p(x)$ 已经不是一个概率密度函数，因此不能使用最大似然进行训练。)
但是，由于我们不知道 ground truth $\triangledown_xlog\ p(x)$，导致损失函数 $L_\theta$ 无法直接计算。
为了解决这个问题，本文便采用了 <b>score matching</b> 的方法将损失函数进行转化，使得其不包含 $\triangledown_xlog\ p(x)$。
经过 score matching 转化后的损失函数<b>无需使用对抗学习优化</b>就可以直接在数据集 $D$ 上进行估计，并使用随机梯度下降进行优化，
类似于训练基于似然的模型的对数似然损失函数(具有已知的 normalizing constant $Z_\theta$)。
具体的推导转化如下：</p>

$$\begin{align}L_\theta = \mathbb{E}_{p(x)}[||\triangledown_xlog\ p(x) - s_\theta(x)||_2^2]  & = 2 * \dfrac{1}{2} \mathbb{E}_{p_{data}(x)}[||\triangledown_xlog\ p_{data}(x) - \triangledown_xlog\ p_\theta(x)||_2^2] \\ &  = 2 * \dfrac{1}{2} \mathbb{E}_{p_{data}(x)}[(\triangledown_xlog\ p_{data}(x) - \triangledown_xlog\ p_\theta(x))^2] \\ & = 2 * \dfrac{1}{2} \int p_{data}(x)(\triangledown_xlog\ p_{data}(x) - \triangledown_xlog\ p_\theta(x))^2 dx \\ & = 2  *(\underset{const}{\underbrace{\int \dfrac{1}{2} p_{data}(x)(\triangledown_xlog\ p_{data}(x))^2 dx}} + \underset{(formula\ 1)}{\underbrace{\int \dfrac{1}{2} p_{data}(x)(\triangledown_xlog\ p_{\theta}(x))^2 dx}} - \underset{(formula\ 2)}{\underbrace{\int p_{data}(x)\triangledown_xlog\ p_{\theta}(x)\triangledown_xlog\ p_{data}(x) dx)}} \end{align}$$

$$formula\ 1:\ \int \dfrac{1}{2} p_{data}(x)(\triangledown_xlog\ p_{\theta}(x))^2 dx = \dfrac{1}{2} \mathbb{E}_{p_{data}}[(\triangledown_xlog\ p_{\theta}(x))^2]$$

$$\begin{align} formula\ 2(分部积分):\ & - \int p_{data}(x)\triangledown_xlog\ p_{\theta}(x)\triangledown_xlog\ p_{data}(x) dx \\ = & - \int \triangledown_xlog\ p_{\theta}(x)\triangledown_x\ p_{data}(x) dx \\ = & -p_{data}(x)\triangledown_xlog\ p_{\theta}(x)|^\infty_{-\infty} + \int p_{data}(x)\triangledown_x^2log\ p_{\theta}(x)dx \\ \overset{(i)}{=} &\ \mathbb{E}_{p_{data}}[\triangledown_x^2log\ p_{\theta}(x)] \Leftarrow |x| \rightarrow 0, p_{data}(x) \rightarrow 0 \end{align}$$

$$\begin{align}Final:\ L_\theta = \mathbb{E}_{p(x)}[||\triangledown_xlog\ p(x) - s_\theta(x)||_2^2]  = 2\ \mathbb{E}_{p_{data}}[\triangledown_x^2log\ p_{\theta}(x)] + \mathbb{E}_{p_{data}}[(\triangledown_xlog\ p_{\theta}(x))^2] + const\end{align}$$

<p style="text-align:justify; text-justify:inter-ideograph;">可以看到，此时的损失函数 $L_\theta$ 不包含 $\triangledown_xlog\ p(x)$。因此可以使用数据集 $D$ 对模型 $s_\theta(x) \approx \triangledown_xlog\ p_{\theta}(x)$ 进行训练
(注意：$\triangledown_x^2log\ p_{\theta}(x) \approx \triangledown_xs_{\theta}(x)$)。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">在训练完成后，由于我们不是直接使用模型训练来拟合 $p(x)$，因此不能对模型进行直接采样生成数据。
为了解决这个问题，本文采用了一个迭代过程，称为 <b>Langevin dynamics</b> 来使用模型 $s_\theta(x) \approx \triangledown_xlog\ p_{\theta}(x)$ 进行采样生成数据。
具体而言，Langevin dynamics 提供了一个 MCMC 迭代过程(链式迭代)，使得可以仅使用一个分布 $p(x)$ 的得分函数 $\triangledown_xlog\ p(x)$ 就可以实现数据的采样。
在迭代开始前，它从任意的先验分布 $x_0 \sim \pi(x)$ 初始化链，然后使用如下的迭代方程进行数据 $x$ 的更新：</p>

<center>$$x_{i+1} \leftarrow x_i + \epsilon \triangledown_xlog\ p(x) + \sqrt{2\epsilon} z_i\underset{\Downarrow}{,} i = 0,1,...,K, z_i \sim \mathcal{N}(0,I) \\ x_{i+1} \leftarrow x_i + \epsilon s_\theta(x) + \sqrt{2\epsilon} z_i$$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">当 $\epsilon \rightarrow 0$，$K \rightarrow \infty$ 时, 迭代更新得到的 $x_K$ 收敛为 $p(x)$ 的某个采样数据。
而在具体实践中，只要 $\epsilon$ 很小，$K$ 很大，则生成的 $x_K$ 与真实采样数据(也就是 $\epsilon \rightarrow 0$，$K \rightarrow \infty$ 的理论采样数据)的误差就可以忽略不计。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">这样，拟合目标的选择，模型的训练和数据的采样推理过程就都可以实现了。<b>总结而言</b>，就是使用 score function $\triangledown_xlog\ p(x)$ 作为训练的拟合目标；
然后设计一个模型 $s_\theta(x)$ 去拟合它：$s_\theta(x) \approx \triangledown_xlog\ p_{\theta}(x)$；接着使用 score mathch 目标函数来训练模型；
最后设置固定的 $\epsilon$ 和 $K$，使用 Langevin dynamics 的迭代采样过程生成数据 $x$。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">但是，上述的流程存在一些缺点。
其中，最关键的不足是，估计的 score function 在低密度区域(即 $p(x)$ 较小的区域)是不准确的，因为在那里只有很少的数据点可用于计算 score matching 目标函数。
而且这是不可避免的，因为 score matching 是通过 minize Fisher divergence 来训练模型：</p>

<center>$$\mathbb{E}_{p(x)}[||\triangledown_xlog\ p(x) - s_\theta(x)||_2^2] = \int p(x) ||\triangledown_xlog\ p(x) - s_\theta(x)||_2^2 dx$$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">可以看到，score-based model $s_\theta(x)$ 由 $p(x)$ 加权，它们在 $p(x)$ 很小的低密度区域很大程度上被忽略。这就导致其在低密度区域的预测不准确。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">为此，本文提出使用 <b>multiple noise perturbations</b> 来改进。其基本思想是用噪声干扰数据点(即在数据点上加噪声)，然后使用含有噪声的数据点训练 score-based model。
当噪声幅度足够大时，它可以填充低数据密度区域，以提高 score function 的准确性。同时，不同的噪声幅度对数据的影响不同：
较大的噪声显然可以覆盖更多的低密度区域，以获得更好的分数估计，但它过度破坏了数据，并显著改变了原始分布。而较小的噪声对原始数据分布的破坏较小，但不能像我们希望的那样尽可能覆盖低密度区域。
因此，本文同时使用多个尺度的噪声对数据进行扰动。
具体而言，假设我们添加的是高斯噪声，首先选择 $L$ 个单调递增的标准差 $\sigma_i: \sigma_1 < ... < \sigma_L$。
然后使用均值 $\mu = 0$，标准差 $\sigma = \sigma_i$ 的 $L$ 个高斯分布 $\mathcal{N}(0, \sigma^2_iI), i=1,...,L$ 来扰动原始的数据分布 $p(x)$，获得 $L$ 个含有噪声的 noise-perturbed distribution：</p>

<center>$p_{\sigma_i}(x) = \int p(y)\mathcal{N}(x;y,\sigma_i^2I)dy$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;">在具体的实现上，我们可以简单地采样一个噪声 $z \sim \mathcal{N}(0,I)$ 和一个原始数据 $x \sim p(x)$，然后将其加权相加即可得到含有噪声的数据 $x_{\sigma_i} = x + \sigma_iz$。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">接着便可以使用模型来估计各个 noise-perturbed distribution $p_{\sigma_i}(x)$ 的 score function $\triangledown_xlog\ p_{\sigma_i}(x)$。
由于此时训练时包含多个数据分布 $p_{\sigma_i}(x)$，因此，需要对原始模型 $s_\theta(x)$ 进行改进，使其对当前的数据分布的噪声程度有所了解。
具体而言，本文使用最简单的方式，即将噪声的程度 $i$ 作为输入和数据 $x$ 一起输入进模型进行学习来估计不同的含噪声的数据分布的 score function：$s_\theta(x,i) \approx \triangledown_xlog\ p_{\sigma_i}(x), i=1,...,L$，
得到 $s_\theta(x,i)$，称为 <b>Noise Conditional Score-Based Model</b>。
其具体训练损失函数相比于原始的损失函数多了一个对所有的数据分布的加权求和，即 $s_\theta(x,i)$ 的训练目标是所有噪声尺度下 Fisher divergences 的加权和：</p>

<center>$$\begin{align}\mathbb{E}_{p_{\sigma_i}(x)}[||\triangledown_xlog\ p_{\sigma_i}(x) - s_\theta(x, i)||_2^2] & = 2\ \mathbb{E}_{p_{data, \sigma_i}}[tr(\triangledown_x^2log\ p_{\theta}(x, i)) + \dfrac{1}{2}||\triangledown_xlog\ p_{\theta}(x, i)||_2^2] +const \\ & = 2\ \mathbb{E}_{p_{data, \sigma_i}}[tr(\triangledown_xs_{\theta}(x, i)) + \dfrac{1}{2}||s_{\theta}(x, i)||_2^2] + const \end{align}$$</center>

<center>$$\mathbb{E}_{p(x)}[||\triangledown_xlog\ p(x) - s_\theta(x,i)||_2^2] = \sum_{i=1}^L\lambda(i)\mathbb{E}_{p_{\sigma_i}(x)}[||\triangledown_xlog\ p_{\sigma_i}(x) - s_\theta(x, i)||_2^2], \lambda(i) \in \mathbb{R}_{>0}, \lambda(i) = \sigma_i^2$$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">然后和原始的方法一样，可以直接使用 score matching 进行转化训练。在训练完成后，由于 $i$ 越小，表示噪声越少，
则最简单的采样过程是使用 $s_\theta(x,1) \approx \triangledown_xlog\ p_\theta(x)$，然后初始化 $x_0 \sim \pi(x)$，并使用 Langevin dynamics 进行迭代更新。
但是其他分布的 score function 就被遗弃了，没有将它们充分利用。而本文使用了一个更好的 Langevin dynamics 迭代方法(称为 <b>annealed Langevin dynamics</b>)，使得可以充分利用各个分布的 score function。
具体算法流程如下图：具体而言，首先初始化 $\bar{x}_0 \sim \pi(x)$，然后将 $\sigma_i$ 从大到小进行循环：$\sigma_L > ... > \sigma_1$。对于每个 $\sigma_i, i = L,..,1$：
先计算 $\alpha_i = \epsilon · \sigma^2_i / \sigma^2_L$ (称为 step size，和 $\epsilon$ 的作用一致，只是加上了权重)；
接着循环迭代步骤 $t = 1,...,T$ (T 和 K 一样，只是符号表示问题)。对于每个迭代步骤，使用 Langevin dynamics 迭代公式进行更新：</p>

<center>$\bar{x}_{t} \leftarrow \bar{x}_{t-1} + \dfrac{\alpha_i}{2} s_\theta(\bar{x}_{t-1}, i) + \sqrt{\alpha_i} z_t, z_t \sim \mathcal{N}(0,I)$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;">最后，<b>将 $\bar{x}_0$ 更新为 $\bar{x}_{T}$：$\bar{x}_{0} = \bar{x}_{T}$</b>。进行下一次循环 $\sigma_{i+1}$。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">这里，除了 $\sigma_L$ 的迭代过程是使用初始化 $\bar{x}_0 \sim \pi(x)$ 进行生成数据，其余的 $\sigma_L$ 的迭代过程都是使用前一次生成的数据作为初始化进行进一步更新，
这样，不仅可以利用每一个分布生成的数据，更减少了每一个分布生成数据的迭代次数。</p>

<p style="text-align:justify; text-justify:inter-ideograph;"><b>总结而言</b>，首先，使用 $L$ 个不同程度的噪声分别扰动原始数据的分布，获得 $L$ 个含噪分布；
然后使用每个含噪分布的 score function $\triangledown_xlog\ p_{\sigma_i}(x)$ 作为训练的拟合目标；
并设计一个模型 $s\theta(x, i)$ 去拟合它：$s_\theta(x, i) \approx \triangledown_xlog\ p_{\theta,\sigma_i}(x)$；接着使用 score mathch 目标函数来训练模型；
最后设置固定的 $\epsilon$ 和 $T$，使用 annealed Langevin dynamics 的迭代采样过程生成数据 $x$。</p>

![annealed Langevin dynamics](/images/score-based_generative_model-annealed_Langevin_dynamics.png)

<p style="text-align:justify; text-justify:inter-ideograph;">到目前为止，我们已经成功实现了通过加入<b>离散的</b>噪声以实现数据的生成。接下来， 我们将其推广到加入<b>连续的</b>噪声。
连续噪声的情况很简单：当噪声的数量 $L$ 接近无穷大时，其本质上就是用不断增长的噪声水平来扰动数据分布。
在这种情况下，噪声扰动过程是一个连续时间随机过程(continuous-time stochastic process)，它是随机微分方程(stochastic differential equations (SDE))的解。
而常见的 SDE 的迭代方程为：</p>

<center>$dx = f(x,t)dt + g(t)dw, f(·，t):\mathbb{R}^d \rightarrow \mathbb{R}^d, g(t) \in \mathbb{R} \Rightarrow dx = e^tdw$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;">其中，$f(·，t)$ 称为 drift coefficient, $g(t)$ 称为 diffusion coefficient, $w$ 表示一个 standard Brownian motion(标准布朗运动)。
而 SDE 的解是一个数据变量的连续集合 $\{x(t)\}_{t \in [0,T]}$ (也就是加入连续噪声生成的数据分布)。这些变量是沿着时间 index $t$ 从开始时间 $0$ 增长到结束时间 $T$ 的随机轨迹。
这里我们使用 $p_t(x)$ 表示 $x(t)$ 的(边际)概率密度函数，则 $p_0(x) = p(x), p_T(x) \approx \pi(x)$ 称为 prior distribution (和离散情况的 $p_{\sigma_L}(x)$ 相似)。
和离散的情况不同(离散情况是通过间接设计 $\sigma_i$ 来控制噪声)，SDE 可以通过直接设计等式方程 $dx = f(x,t)dt + g(t)dw$ 来控制噪声。
例如，方程 $\boldsymbol{dx = e^tdw}$ 表示使用均值 $\mu=0$ 且方差 $\sigma^2$ 指数增长的高斯噪声扰动数据，(即 $f(x,t) = 0, g(t) = e^t$)。
注意，这里 SDE 的等式方程是和离散情况的 $x_{\sigma_i} = x + \sigma_iz$ 类似，表示对数据进行加噪。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">而在有了噪声数据后，和离散情况类似，我们可以训练模型预测分布 $p_t(x)$ 的 score function 来进行训练：$s_\theta(x,t) \approx \triangledown_xlog\ p_t(x)$，称为 <b>Time-Dependent Score-Based Model</b>。
不同的是，这里使用的模型 $s_\theta(x,t)$ 的输入 $t$ 是实数，即 $t \in \mathbb{R}^+$。而训练损失函数则也是 Fisher divergences 的加权和：</p>

<center>$$\begin{align} & \mathbb{E}_{t \sim \mathcal{U}(0,T)}\mathbb{E}_{p_t(x)}[\lambda(t)||\triangledown_xlog\ p_t(x) - s_\theta(x,t)||_2^2], \lambda: \mathbb{R} \rightarrow \mathbb{R}_{>0}, \lambda(t) \propto \dfrac{1}{\mathbb{E}[||\triangle_{x(t)log\ p(x(t))|x(0)}||_2^2]} \\ = & \mathbb{E}_{t \sim \mathcal{U}(0,T)} 2\ \mathbb{E}_{p_{t}(x)}[tr(\triangledown_x^2log\ p_t(x)) + \dfrac{1}{2}||\triangledown_xlog\ p_t(x)||_2^2] +const \\ = & 2\ \mathbb{E}_{t \sim \mathcal{U}(0,T)}\mathbb{E}_{p_{t}(x)}[tr(\triangledown_xs_{\theta}(x, t)) + \dfrac{1}{2}||s_{\theta}(x, t)||_2^2] + const \end{align}$$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">其中，$\mathcal{U}(0,T)$ 表示时间间隔 $[0,T]$ 上的均匀分布，
而 $\lambda(t) \propto \dfrac{1}{\mathbb{E}[||\triangle_{x(t)log\ p(x(t))|x(0)}||_2^2]}$ 用来平衡不同 score matching 的损失的大小。
然后将其转化为 score matching 进行训练。而在完成训练后，在离散情况下是使用 annealed Langevin dynamics 进行迭代采样，而在 SDE 下，本文使用 使用 <b>reverse SDE</b> 来反向样本生成的扰动过程，
即首先初始化 $x(T) \sim \pi$，然后使用 reversal SDE 迭代得到 $x(0)$。值得注意的是，每一个 SDE 都存在对应的 reversal SDE，其方程形式如下：</p>

<center>$dx = [f(x,t) - g^2(t)\triangledown_xlog\ p_t(x)]dt + g(t)dw \approx [f(x,t) - g^2(t)s_\theta(x,t)]dt + g(t)dw$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;">其中，$dt$ 表示<b>负的</b>无穷小时间步长，且 reversal SDE 的迭代过程是从大到小的，即从 $t=T$ 到 $t=0$。
在迭代过程的具体实现上，通过使用 numerical SDE solver 求解预测的 reversal SDE，就可以模拟样本生成的反向随机过程。
例如对于 Euler solver (Euler-Maruyama)，它使用有限时间步长和小高斯噪声来将 SDE 离散化。
它选择一个较小的负时间步长 $t \approx 0$，并初始化 $t = T$，$x = x(T) \sim \pi$，然后使用如下迭代更新公式，直到 $t=0$：</p>

<center>$$\triangle x \rightarrow [f(x,t) - g^2(t)s_\theta(x,t)]\triangle t + g(t) \sqrt{|\triangle t|}z_t, x \rightarrow x + \triangle x, t \rightarrow t + \triangle t, \triangle t < 0, \triangle t \approx 0, z_t \sim \mathcal{N}(0,I)$$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">可以看到，Euler-Maruyama 方法类似于 Langevin dynamics，都是通过使用高斯噪声扰动的 score function 来更新 $x$。这样，我们也实现了加入连续的噪声以实现数据的生成。
<b>总结而言</b>，首先，使用 SDK 的迭代方程来扰动原始数据的分布，获得 $[0,T]$ 区间内的含噪分布；
然后使用 $[0,T]$ 区间内的所有含噪分布的 score function $\triangledown_xlog\ p_{t}(x)$ 作为训练的拟合目标；
并设计一个模型 $s\theta(x, t)$ 去拟合它：$s_\theta(x,t) \approx \triangledown_xlog\ p_{\theta,t}(x)$；接着使用 score mathch 目标函数来训练模型；
最后使用 reversal SDE 的 numeral SDE solver 迭代采样过程生成数据 $x$。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">但是，无论是基于 Langevin MCMC 的离散采样过程，还是基于 SDE solver 的连续采样过程，它们都没有提供一种 score-based generative models 的<b>精确的</b>对数似然的方法
(都只是拟合了 score function，无法得到 $p(x)/log\ p(x)$)。为了解决这个问题，本文引入一个基于常微分方程(ordinary differential equation， ODE)的采样器，它能够实现精确的似然计算。
值得注意的是，任意一个 SDE 都可以转化为一个 ODE (称为 probability flow (PE) ODE)，同时保持其边缘分布 $\{p_t(x)\}_{t \in [0,T]}$ 不变。其(ODE)具体方程公式如下：

<center>$$dx = [f(x,t) - \dfrac{1}{2}g^2(t)\triangledown_xlog\ p_t(x)]dt$$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">通过求解这个 ODE，可以获得和求解 reversal SDE 得到的相同的边缘分布 $\{p_t(x)\}_{t \in [0,T]}$。
也就是说，求解 ODE 和 求解 reversal SDE 得到的结果是相同的。但是，相比于 reversal SDE，ODE 有如下优点：</p>

<p style="text-align:justify; text-justify:inter-ideograph;">当 score function $\triangledown_xlog\ p(x)$ 使用它的估计模型 $s_\theta(x,t)$ 替换后，PE ODE 就成为了 neural ODE 的一个特例。
具体而言，它是 continuous normalizing flows 的一个特例，因为 PE ODE 将数据分布 $p_0(x)$ 转换为先验噪声分布 $p_T(x)$ (它与SDE共享相同的边缘分布 $\{p_t(x)\}_{t \in [0,T]}$)，并且是完全可逆的。
因此，PE ODE 继承了 neural ODE / continuous normalizing flows 的所有性质，包括精确的对数似然计算。
具体来说，我们可以利用瞬时变量变化公式并借助 numerical ODE solver 从已知的先验噪声分布/先验密度 $p_T$ 计算未知的数据分布/数据密度 $p_0$。其中瞬时变量变化公式( instantaneous change-of-variable formula)如下：</p>

<center>$$\underset{solutions}{\underbrace{\left[ \begin{array}{c} z_0 \\ log\ p(x) - log\ p_{z_0}(z_0) \end{array}\right]}} = \left[ \begin{array}{c} x \\ 0 \end{array}\right] + \underset{dynamics}{\underbrace{\int_{t_1}^{t_0} \left[ \begin{array}{c} f(z(t),t;\theta), Tr(\dfrac{\partial f}{\partial z(t)}) \end{array}\right]}} dt \\ \underset{inital\ values}{\underbrace{\left[ \begin{array}{c} z(t_1) \\ log\ p(x)-log\ p(z(t_1)) \end{array}\right] = \left[ \begin{array}{c} x \\ 0 \end{array}\right]}}$$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">除了能通过不同的加噪/采样方式获得更好的图像质量，score-based generative model 还能解决 inverse problem (逆问题，即基于条件的图像生成)。
因为本质上，逆问题与贝叶斯推理问题相同。假设 $x$ 和 $y$ 是两个随机变量，其中 $y$ 是条件，$x$ 是任务，
且我们知道从 $x$ 生成 $y$ 的正向过程，由转移概率分布 $p(y|x)$ 表示。那么反向问题/逆问题就是根据条件 $y$ 计算条件概率 $p(x|y)$。
求解方式如下：通过贝叶斯规则，并将两边同时取关于 $x$ 的梯度，可以大大简化贝叶斯表达式，并得到以下 score function 的贝叶斯规则：</p>

<center>$p(x|y) = \dfrac{p(x)p(y | x)}{\int p(x)p(y|x)dx} \Rightarrow \triangledown_xlog\ p(x|y) = \triangledown_xlog\ p(x) + \triangledown_xlog\ p(y|x), s_\theta(x) \approx \triangledown_xlog\ p(x|y)$</center>

<p style="text-align:justify; text-justify:inter-ideograph;"></p>

<p style="text-align:justify; text-justify:inter-ideograph;">通过 score matching，我们可以训练一个模型 $s_\theta(·)$ 来估计无条件数据分布(即 $p(x)$)的 score function $s_\theta(x) \approx \triangledown_xlog\ p(x)$。
且由于 $p(y|x)$ 是已知的，我们就可以通过上述方程从已知的前向过程 $p(y|x)$ 中轻松计算后验 score function $\triangledown_xlog\ p(x|y)$，并使用 Langevin-type sampling 从中采样，
最终生成基于条件 $y$ 下的图像 $x$。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">例如，使用无条件的 time-dependent score-based model $s_\theta(x,t)$ 和预训练的噪声条件图像分类器 $p(y|x)$，就可以实现基于类别的图像生成，其中 $y$ 是类标签，$x$ 是图像。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">最后，虽然 score-based generative model 和 <a href="https://cai-jianfeng.github.io/posts/2023/11/blog-diffusion-model/" target="_blank">DM 模型</a> 的推理角度不同，
但是本质上，通过将噪声的数量扩展到无穷大(即 SDE 情况/ ODE 情况下)，可以证明 score-based generative model 和 DM 模型都可视为由 score function 确定的 SDE 的离散形式
(在 score-based generative model 是使用离散(高斯)噪声 + Langevin dynamics 迭代采样；而在 DM 模型中是使用离散(高斯)噪声 + 正态分布迭代采样)，
这样就将 score-based generative model 和 DM 模型连接到一个统一的框架中。</p>

<h1>代码实现(Pytorch)</h1>

<h1>附录</h1>

<p style="text-align:justify; text-justify:inter-ideograph;">1. 常见的生成模型可以分成 $2$ 个类别：<b>likelihood-based models</b> 和 <b>implicit generative models</b>。
其中 likelihood-based models 通过(近似)最大似然的目标函数直接学习分布的概率密度(或质量)函数。
典型的模型包括 autoregressive models、normalizing flow models、energy-based models (EBMs) 和 variational auto-encoders (VAEs)。
本文的 score-based generative model 也属于这类模型。
而 implicit generative models 的概率分布是由抽样过程的模型隐式表示。
最典型的例子是生成式对抗网络(GANs)，其中通过用神经网络转换随机高斯向量来合成数据分布中的新样本。

<p style="text-align:justify; text-justify:inter-ideograph;">然而，likelihood-based models 和 implicit generative models 都有显著的局限性。
likelihood-based models 要么需要对模型架构进行严格的限制，以确保似然计算的可处理的 normalizing constant，要么必须依赖代理目标来近似最大似然训练。
而 implicit generative models 通常需要对抗性训练，众所周知，这种训练是不稳定的，并可能导致模型坍塌。

<p style="text-align:justify; text-justify:inter-ideograph;">2. 使用 score-based generative models with multiple noise perturbations 的实用建议：</p>

<ul><li>
<p style="text-align:justify; text-justify:inter-ideograph;">选择 $\sigma_1 < ... < \sigma_L$ 作为几何级数，并且 $\sigma_1$ 足够小，$\sigma_L$ 可与所有训练数据点之间的最大成对距离相比较。$L$ 通常是几百或几千的数量级。</p>
</li>
<li>
<p style="text-align:justify; text-justify:inter-ideograph;">使用 U-Net skip connections 对  score-based model $s_\theta(x,i)$ 进行参数化，也就是使用 U-Net with skip connections 作为 $s_\theta(x,i)$。</p>
</li>
<li>
<p style="text-align:justify; text-justify:inter-ideograph;">在测试时，对 score-based model 的权重应用指数移动平均(EMA)。</p>
</li></ul>

<p style="text-align:justify; text-justify:inter-ideograph;">3. 在 reverse SDE 的情况下，当 $\lambda(t) = g^2(t)$ 时，在一定的正则条件下，Fisher divergences 的加权组合与 $p_0$ 和 $p_\theta$ KL divergence 有重要的联系：</p>

<center>$$\lambda(t) = g^2(t) \Rightarrow KL(p_0(x)||p_\theta(x)) \leq \dfrac{T}{2} \mathbb{E}_{t \sim \mathcal{U}(0,T)}\mathbb{E}_{p_t(x)}[\lambda(t)||\triangledown_xlog\ p_t(x) - s_\theta(x,t)||_2^2] + KL(p_T|| \pi)$$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">由于这种与 KL divergence 的特殊联系，以及最小化 KL divergence 和最大化似然之间的等价性，我们称 $\lambda(t) = g^2(t)$ 为似然加权函数。
使用这个似然加权函数，我们可以训练基于分数的生成模型，以实现非常高的似然，与 SOTA 的 autoregressive models 相当甚至更好。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">4. reverse SDE 的进一步改进：reverse SDE 有两个特殊的特性，能够更灵活的采样方法：</p>

<ul><li>
<p style="text-align:justify; text-justify:inter-ideograph;">我们通过 time-dependent score-based model $s_\theta(x,t)$ 来估计 $\triangledown_xlog\ p_t(x)$。</p>
</li>
<li>
<p style="text-align:justify; text-justify:inter-ideograph;">我们只关心从每个边缘分布 $p_t(x)$ 中采样。在不同时间步长的样本可以具有任意的相关性，而不必形成从 reverse SDE 中采样的特定轨迹。</p>
</li></ul>

<p style="text-align:justify; text-justify:inter-ideograph;">由于这两个性质，我们可以应用 MCMC 方法来微调从 numerical SDE solvers 中获得的轨迹。
具体来说，本文提出了预测-校正采样器(Predictor-Corrector samplers)。
其中，Predictor 可以是任意的 numerical SDE solver，它从现存的样本 $x(t) \sim p_t(x)$ 预测 $x(t+\triangle t) \sim p_{t+\triangle t}(x)$，例如 Euler solver。
而 Corrector 可以是任意的完全依赖于 score function 的 MCMC 过程，例如离散情况下的 Langevin dynamics。
在 Predictor-Corrector samplers 的每一步，我们首先使用 Predictor 选择合适的步长 $\triangle t < 0$，然后基于当前样本 $x(t)$ 预测 $x(t+\triangle t)$：</p>

<center>$$\triangle x \rightarrow [f(x,t) - g^2(t)s_\theta(x,t)]\triangle t + g(t) \sqrt{|\triangle t|}z_t, x \rightarrow x + \triangle x, t \rightarrow t + \triangle t$$</center>

<p style="text-align:justify; text-justify:inter-ideograph;">接下来，我们运行几个 Corrector 步骤，根据score-based model $s_\theta(x,t +\triangle t) $改进样本 $x(t+\triangle t)$，
使 $x(t+\triangle t)$ 成为 $p_{t+\triangle t}(x)$ 的高质量样本。</p>

<center>$x(t+\triangle t) \leftarrow x(t+\triangle t) + \epsilon \triangledown_xlog\ p_\theta_{t +\triangle t}(x) + \sqrt{2\epsilon} z, z \sim \mathcal{N}(0,I) \\ \\ x(t+\triangle t) \leftarrow x(t+\triangle t) + \epsilon s_\theta(x, t+\triangle t) + \sqrt{2\epsilon} z$</center>