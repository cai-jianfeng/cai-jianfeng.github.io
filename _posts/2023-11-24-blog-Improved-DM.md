---
title: 'The Advanced Knowledge of Diffusion Model (DM)'
date: 23-11-24
permalink: /posts/2023/11/blog-improved-diffusion-model/
tags:
  - 深度学习基础知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客参考了<a href="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/" target="_blank">
What are Diffusion Models?</a>，继续详细讲述了最近大火的 DM 模型的改进的数学原理/推导及编程
(ps：DM 的基础知识详见 <a href="https://cai-jianfeng.github.io/posts/2023/11/blog-diffusion-model/" target="_blank">The Basic Knowledge of Diffusion Model (DM)</a>)。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">注意：公式中的$\color{Green}{绿色}$字体公式表示在 <a href="https://cai-jianfeng.github.io/posts/2023/11/blog-diffusion-model/" target="_blank">The Basic Knowledge of Diffusion Model (DM)</a> 中已经推理得到了。
$\color{Red}{红色}$字体公式表示使用$\color{Green}{绿色}$字体的公式进一步推理得到的/从附录中推理得到的。

<h1>DDIM</h1>

<p style="text-align:justify; text-justify:inter-ideograph;">回顾 <a href="https://cai-jianfeng.github.io/posts/2023/11/blog-diffusion-model/" target="_blank">The Basic Knowledge of Diffusion Model (DM)</a>，$x_{t-1}$ 可以由如下方程推导：</p>

$$\begin{align}x_{t-1} & = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\bar{\varepsilon}_{t-1}, \color{Green}{\bar{\varepsilon}_{t-1} \sim \mathcal{N}(0, \boldsymbol{I})} \\ 
& = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}\bar{\varepsilon}_{t} + \sigma_t^2\bar{\varepsilon}, \sigma_t^2 = \eta \dfrac{\beta_t(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \\ 
& = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}\color{Red}{\dfrac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}}} + \sigma_t^2\bar{\varepsilon} \leftarrow \color{Green}{x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\bar{\epsilon}_t} \end{align}$$

<p style="text-align:justify; text-justify:inter-ideograph;">(在之前的推理时 $\eta = 1$，这里我们将其扩展，使用一个变量 $\eta$ 来控制方差)根据上述公式，可以得到 $q_\sigma(x_{t-1}|x_t,x_0)$ 的分布：</p>

$$q_\sigma(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1};\sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}\dfrac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma^2_tI)$$

<p style="text-align:justify; text-justify:inter-ideograph;">令 $\eta = 0$，就可以获得 <b>DDIM</b> 的第一个改进，即随机方差 $\sigma^2=0$：</p>

$$q_\sigma(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1};\sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\dfrac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}}, 0)$$

<p style="text-align:justify; text-justify:inter-ideograph;">同时令 $\{\tau_1,...,\tau_S\}, \tau_1 < ... < \tau_S \in [1, T], S < T$，就可以获得 <b>Improved DDPM</b> 的改进，即从 $[1,T]$ 抽样部分步骤完成逆扩散过程：</p>

$$q_{\sigma, \tau}(x_{\tau_{i-1}}|x_{\tau_i},x_0) = \mathcal{N}(x_{\tau_{i-1}};\sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}\dfrac{x_{\tau_i} - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma^2_tI)$$ 

<p style="text-align:justify; text-justify:inter-ideograph;">将两者结合，就得到了 <b>DDIM</b>：</p>

$$q_{\sigma, \tau}(x_{\tau_{i-1}}|x_{\tau_i},x_0) = \mathcal{N}(x_{\tau_{i-1}};\sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\dfrac{x_{\tau_i} - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}}, 0)$$

$$x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\dfrac{x_{\tau_i} - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}} + 0 \times \epsilon_{\tau_{i-1}}$$

<p style="text-align:justify; text-justify:inter-ideograph;">由于 $x_0$ 是未知的, 所以给定一个含噪图像 $x_{\tau_i}$, 首先需要预测出对应的 $x_0$, 
然后使用给定的 $x_{\tau_i}$ 和预测得到的 $x_0$ 通过上述的反向条件分布方程 $q_\sigma(x_{\tau_{i-1}} \vert x_{\tau_i},x_0)$ 预测 $x_{\tau_{i-1}}$。
具体而言，首先模型 $\epsilon_\theta(x_{\tau_i})$ 输入含噪图像 $x_{\tau_i}$ 预测噪声 $\epsilon_{\tau_i}$，
然后通过如下方程使用 $x_{\tau_i}$ 和预测的噪声 $\epsilon_{\tau_i}$ 获得 $x_0$：</p>

$$\color{Green}{x_{\tau_i} = \sqrt{\bar{\alpha}_{\tau_i}}x_0 + \sqrt{1 - \bar{\alpha}_{\tau_i}}{\epsilon}_{\tau_i}} \rightarrow  x_0 = \dfrac{1}{\sqrt{\bar{\alpha}_{\tau_i}}}(x_{\tau_i} - \sqrt{1 - \bar{\alpha}_{\tau_i}}{\epsilon}_{\tau_i})$$

<p style="text-align:justify; text-justify:inter-ideograph;">接着将 $x_0$ 代入上述的更新公式(公式 (8))，最终预测得到更新的 $x_{\tau_{i-1}}$

$$\begin{align}x_{\tau_{i-1}} & = \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\dfrac{x_{\tau_i} - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}} \\ 
&  = \sqrt{\bar{\alpha}_{t-1}}\dfrac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}{\epsilon}_t) + \sqrt{1 - \bar{\alpha}_{t-1}}\dfrac{\color{Blue}{x_{\tau_i}} - \color{Purple}{\sqrt{\bar{\alpha}_t}\dfrac{1}{\sqrt{\bar{\alpha}_t}}}(\color{Blue}{x_t} - \color{Orange}{\sqrt{1 - \bar{\alpha}_t}}{\epsilon}_t)}{\color{Orange}{\sqrt{1 - \bar{\alpha}_t}}} \\ & = \sqrt{\bar{\alpha}_{t-1}}(\dfrac{x_t - \sqrt{1 - \bar{\alpha}_t}{\epsilon}_t}{\sqrt{\bar{\alpha}_t}}) + \sqrt{1 - \bar{\alpha}_{t-1}}\epsilon_t \end{align}$$

<h1>Condition</h1>

<h2>Classifier Guidance</h2>

<p style="text-align:justify; text-justify:inter-ideograph;">可以训练一个分类器 $f_\phi(y \vert x_t,t)$，
其将含噪图像 $x_t$ 分类为其类别 $y$，然后使用分类器对于输入(含噪图像) $x_t$ 的梯度 $\nabla_{x_t}log\ f_\phi(y \vert x_t)$ 来指导模型的逆扩散过程，
使其偏移到基于条件 $y$ 进行逆扩散生成图像的样本空间。
具体而言，假设无条件的样本空间为 $q(x)$，则基于条件 $y$ 的样本空间则为 $q(x,y)$，
由于 $q(x)/q(x,y)$ 都服从高斯分布，因此其梯度为 $-\dfrac{\epsilon}{\sigma}$ (详细推导见附录 A)，然后通过贝叶斯规则的转化，即可实现从无条件到有条件的转化：</p>

$$\begin{align}\nabla_{x_t}log\ q(x_t,y) & = \nabla_{x_t}log\ q(x_t) + \nabla_{x_t}log\ q(y|x_t) \\ & \approx \nabla_{x_t}log\ q(x_t) + \nabla_{x_t}log\ f_\phi(y|x_t) \leftarrow \color{Red}{\nabla_{x_t}log\ q(x_t) = - \dfrac{1}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta(x_t,t)} \\ 
& = - \dfrac{1}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta(x_t,t) + \nabla_{x_t}log\ f_\phi(y|x_t) \\ 
& = - \dfrac{1}{\sqrt{1 - \bar{\alpha}_t}}(\epsilon_\theta(x_t,t) - \sqrt{1 - \bar{\alpha}_t}\nabla_{x_t}log\ f_\phi(y|x_t)) \end{align}$$ 

<p style="text-align:justify; text-justify:inter-ideograph;">因此，在使用给定分类器对于 $x_t$ 的梯度对无条件情况下预测得到的噪声 $\epsilon_\theta(x_t,t)$ 进行指导(即相加)，就可以得到基于条件 $y$ 下的噪声 $\bar{\epsilon}_\theta(x,t)$：</p>

$$\bar{\epsilon}_\theta(x,t) = \epsilon_\theta(x_t,t) - \sqrt{1 - \bar{\alpha}_t}\nabla_{x_t}log\ f_\phi(y|x_t)$$

<p style="text-align:justify; text-justify:inter-ideograph;">为了控制分类器梯度指导的强度，可以给梯度部分添加一个权重 $\omega$ 来控制：</p>

$$\bar{\epsilon}_\theta(x,t) = \epsilon_\theta(x_t,t) - \sqrt{1 - \bar{\alpha}_t}\omega\nabla_{x_t}log\ f_\phi(y|x_t)$$

<h2>Classifer-Free Guidance</h2>

<p style="text-align:justify; text-justify:inter-ideograph;">即使没有独立的分类器$f_\phi(·)$，
我们也可以直接通过无条件模型 $p_\theta(x) \rightarrow \epsilon_\theta(x_t,t)$ 和基于条件 $y$ 的模型 $p_\theta(x \vert y) \rightarrow \epsilon_\theta(x_t,t,y)$ 来实现类似 Classifier-Guidance 的效果。
具体而言，我们可以训练一个基于条件 $y$ 的模型 $\epsilon_\theta(x_t,t,y)$，并对条件 $y$ 进行周期性的随机丢弃，使得模型可以学习无条件的情况
(相当于使用一个模型即学习到了基于条件 $y$ 的情况，又学习到了无条件的情况)，
即 $\epsilon_\theta(x_t,t) = \epsilon_\theta(x_t,t,y=\varnothing)$。这样，在得到了 $\epsilon_\theta(x_t,t)$ 和 $\epsilon_\theta(x_t,t,y)$ 后，
我们就可以使用贝叶斯规则获得类似分类器对于输入(含噪图像) $x_t$ 的梯度：</p>

$$\nabla_{x_t}log\ p(y|x_t) = \nabla_{x_t}log\ p(x_t|y) - \nabla_{x_t}log\ p(x_t) = - \dfrac{1}{\sqrt{1 - \bar{\alpha}_t}}(\epsilon_\theta(x_t,t, y) - \epsilon_\theta(x_t,t))$$

<p style="text-align:justify; text-justify:inter-ideograph;">最后，在获得了分类器对于输入(含噪图像) $x_t$ 的梯度，我们就可以使用 Classifier-Guidance 方法进行更新预测的噪声：</p>

$$\begin{align}\bar{\epsilon}_\theta(x,t,y) & = \epsilon_\theta(x_t,t, y) - \sqrt{1 - \bar{\alpha}_t}\omega\nabla_{x_t}log\ p(y|x_t) \\ & = \epsilon_\theta(x_t,t, y) + \omega(\epsilon_\theta(x_t,t, y) - \epsilon_\theta(x_t,t)) \\ & = (\omega + 1)\epsilon_\theta(x_t,t, y) - \omega\epsilon_\theta(x_t,t) \end{align}$$

<h1>附录</h1>

A. 高斯分布的梯度推导：

$$\begin{align}x \sim \mathcal{N}(\mu,\sigma^2I) & \Rightarrow p(x) = \dfrac{1}{\sqrt{2\pi}\sigma}exp(-\dfrac{(x-\mu)^2}{2\sigma^2}) \\
& \Rightarrow \nabla_xlog\ p(x) = \nabla_x(-\dfrac{(x-\mu)^2}{2\sigma^2}) \\
& \Rightarrow \nabla_xlog\ p(x) = -\dfrac{x-\mu}{\sigma^2} = -\dfrac{\epsilon}{\sigma}, x = \mu + \sigma \times \epsilon \rightarrow \epsilon = \dfrac{x - \mu}{\sigma} \end{align}$$

B. Classifier_guidance 在 DDPM 和 DDIM 框架下的实现：

<p style="text-align:justify; text-justify:inter-ideograph;">DDPM: </p>

$$\begin{align}x_{t-1} & = \dfrac{1}{\sqrt{\alpha_t}}(x_t - \dfrac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\bar{\epsilon}_\theta(x,t)) + \dfrac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t \times z_t, z_t \in \mathcal{N}(0, \boldsymbol{I}) \\ & = \dfrac{1}{\sqrt{\alpha_t}}(\underset{\mu_t}{\underbrace{x_t - \dfrac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta(x_t,t)}} + \sqrt{1 - \bar{\alpha}_t}\omega\nabla_{x_t}log\ f_\phi(y|x_t)) + \underset{\Sigma_t}{\underbrace{\dfrac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t}} \times z_t \end{align}$$

<p style="text-align:justify; text-justify:inter-ideograph;">DDIM: </p>

$$\begin{align}x_{\tau_{i-1}} & \sqrt{\bar{\alpha}_{t-1}}(\dfrac{x_t - \sqrt{1 - \bar{\alpha}_t}\bar{\epsilon}_\theta(x,t)}{\sqrt{\bar{\alpha}_t}}) + \sqrt{1 - \bar{\alpha}_{t-1}}\bar{\epsilon}_\theta(x,t) \\ & = \sqrt{\bar{\alpha}_{\tau_{i-1}}}(\dfrac{x_{\tau_i} - \sqrt{1 - \bar{\alpha}_{\tau_i}}(\epsilon_\theta(x_t,t) - \sqrt{1 - \bar{\alpha}_t}\omega\nabla_{x_t}log\ f_\phi(y|x_t))}{\sqrt{\bar{\alpha}_{\tau_i}}}) + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}}}(\epsilon_\theta(x_t,t) - \sqrt{1 - \bar{\alpha}_t}\omega\nabla_{x_t}log\ f_\phi(y|x_t)) \end{align}$$

<p style="text-align:justify; text-justify:inter-ideograph;">具体而言，它们分别称为消融扩散模型(<b>ADM</b>)和附加分类器指导的模型(<b>ADM-G</b>)：</p>

<img src="https://cai-jianfeng.github.io/images/classifier_guidance.png">

B. Classifier-Guidance 代码框架：由上述推导可知，最后需要将 classifier 的梯度加入到预测的噪声中：

$$\bar{\epsilon}_\theta(x,t) = \epsilon_\theta(x_t,t) - \sqrt{1 - \bar{\alpha}_t}\omega\nabla_{x_t}log\ f_\phi(y \vert x_t)$$

<p style="text-align:justify; text-justify:inter-ideograph;">注意，这里是 classifier 关于输入 $x_t$ 的梯度，而不是 classifier 模型参数的梯度。
因此，我们可以利用 <b>torch</b> 的自动求导机制对 $x_t$ 进行求导，而由于 $x_t$ 的梯度和 $\epsilon_\theta(x_t,t)$ 形状相同(都是原始图像的形状)，
因此我们可以直接将它们进行相加，具体代码框架如下：</p>

<img src="https://cai-jianfeng.github.io/images/classifier_guidance_code.png">

