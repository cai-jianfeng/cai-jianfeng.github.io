DDPM原理
===

方差序列：$\{\beta_{t} \in (0, 1)\}_{t=1}^T$ 

扩散过程：$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1},\beta_t\boldsymbol{I}) \Rightarrow x_t = \sqrt{1 - \beta_t}x_{t-1} + \sqrt{\beta_t}\varepsilon_{t-1}, \varepsilon_{t-1} \in \mathcal{N}(0, 1), \beta_1 < ... < \beta_T \Rightarrow x_T \sim \mathcal{N}(0, \boldsymbol{I})$

逆扩散过程：$q(x_{t-1}|x_t) \Rightarrow x_{t-1} = \dfrac{(x_t - \sqrt{\beta_t}\varepsilon_{t-1})}{\sqrt{1 - \beta_t}} \Rightarrow \varepsilon_{\theta} \approx \varepsilon_t \Rightarrow$ 学习一个网络，使得输入 $x_t$，输出 $\varepsilon_{\theta}$ 

注意：$\beta_{t}, t = 1,...,T$ 对于每一个 $x_0$ 都是固定的；但是 $\varepsilon_t, t = 1,...,T$ 对每个 $x_0$ 都是不固定的，随机采样的，这就导致你想训练数据 $x_t^i \rightarrow x_{t-1}^i$ 时，你只能先扩散 $t$ 步，得到 $\varepsilon_{1:t}^i$，然后才能进行逆扩散过程训练。假设你有 $1B$ 数据，$T$ 通常取 $1000$，为了确保 $1\sim T$ 的每个逆扩散过程都可以充分学习，需要获得每个数据的 $\varepsilon_{1:T}$，也就是在还没开始训练时，准备好训练数据就要计算(扩散) $1B \times 1000$ 次，而且还得存储 $1B \times 1000$ 个 $\varepsilon_t^i$ 的数据，这个代价是很大的。

DDPM 的第一个贡献是，你不需要中间繁琐的 $\varepsilon_{1:t}$，而是用一个 $\bar{\varepsilon}_0$ 就可以通过一步扩散从 $x_0^i$ 到 $x_t^i$，即直接获得 $q(x_t^i|x_0^i)$。这时你需要训练数据 $x_0^i$ 的 $x_t^i \rightarrow x_{t-1}^i$ 时，不需要 $t$ 步扩散得到 $x_t^i$，而是一步扩散就可以得到 $x_t^i$。因此在训练时你也不需要存储 $1B \times 1000$ 个 $\varepsilon_t^i$ 的数据，想训练 $x_t^i \rightarrow x_{t-1}^i$，就直接现场采样 $\bar{\varepsilon}_0$ 并加到 $x_0^i$ 上，就可以获得 $x_t^i$。

回看扩散过程，$x_t = \sqrt{1 - \beta_t}x_{t-1} + \sqrt{\beta_t}\varepsilon_{t-1}$，通过将左边的 $x_{t-1}$ 使用 $x_{t-2}$ 展开，直到 $x_0$，就得到了 $q(x_t|x_0)$，并且仅包含一个随机变量 $\bar{\varepsilon}_0$ 吗？不一定，因为不是所有的概率分布，都可以将任意的 $t$ 随机变量 $\varepsilon_{1:t}$ 融合成一个随机变量 $\bar{\varepsilon}_0$ ，但幸运的是，$\varepsilon_{1:t} \sim \mathcal{N}(0, \boldsymbol{I})$，正态分布，是我见过的最奇妙的概率分布，无论是多个随机变量的加法还是乘法，都可以融合成一个随机变量。这也是为什么扩散模型使用的都是正态分布，而不是其他分布，不仅仅是因为它的常见性，还有它的数学特性，可以帮助简化模型学习。

为了方便，令 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{i=1}^t{\alpha_i}$。

则 $x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1 - \alpha_t}\varepsilon_{t-1}$，

代入 $x_{t-1} = \sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_{t-1}}\varepsilon_{t-2}$ 得

$x_t = \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_{t-1}}\varepsilon_{t-2}) + \sqrt{1 - \alpha_t}\varepsilon_{t-1} = \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{\alpha_t - \alpha_t\alpha_{t-1}}\varepsilon_{t-2} + \sqrt{1 - \alpha_t}\varepsilon_{t-1}$ 

$\sqrt{\alpha_t - \alpha_t\alpha_{t-1}}\varepsilon_{t-2} \sim \mathcal{N}(0, (\alpha_t - \alpha_t\alpha_{t-1})\boldsymbol{I})$，$\sqrt{1 - \alpha_t}\varepsilon_{t-1} \sim \mathcal{N}(0, (1 - \alpha_t)\boldsymbol{I})$，所以 $\sqrt{\alpha_t - \alpha_t\alpha_{t-1}}\varepsilon_{t-2} + \sqrt{1 - \alpha_t}\varepsilon_{t-1} \sim \mathcal{N}(0, (1 - \alpha_t\alpha_{t-1})\boldsymbol{I})$ 也是一个正态分布，用 $\bar{\varepsilon}_{t-2} \sim \mathcal{N}(0, \boldsymbol{I})$ 表示可得

$x_t = \sqrt{\alpha_t\alpha_{t-1}}x_{t-2} + \sqrt{1 - \alpha_t\alpha_{t-1}}\bar{\varepsilon}_{t-2}$

经过不断展开，最终可得 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\bar{\varepsilon}_0, \bar{\varepsilon}_0 \sim \mathcal{N}(0, \boldsymbol{I})$，即 $q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0,(1 - \bar{\alpha}_t)\boldsymbol{I})$

由于 $\beta_{t}, t = 1,...,T$ 是固定的，所以我们可以先计算出每个 $\bar{\alpha}_t$，然后对于需要任意的 $t$ 步扩散数据，只需要现场采样一个 $\bar{\varepsilon}_0$，就可以获得 $x_t$：$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\bar{\varepsilon}_0$

接下来还有一个问题，我们是逆扩散 $x_t \rightarrow x_{t-1}$，预测结果为 $\varepsilon_t$，而不是 $\bar{\varepsilon}_0$，因此还需要进一步将模型转化到预测 $\bar{\varepsilon}_0$。

再回看 $q(x_{t-1}|x_t)$，因为 $x_{t-1}$ 与 $x_0$ 无关，所以可以写成 $q(x_{t-1}|x_t,x_0)$，通过贝叶斯公式分解可得

$\begin{aligned} q(x_{t-1}|x_t,x_0) & = \dfrac{q(x_{t-1},x_t,x_0)}{q(x_t,x_0)} = \dfrac{q(x_t|x_{t-1},x_0) \times q(x_{t-1},x_0)}{q(x_t,x_0)} \\ & = \dfrac{q(x_t|x_{t-1},x_0) \times q(x_{t-1},x_0) / q_{x_0}}{q(x_t,x_0) / q(x_0)} = \dfrac{q(x_t|x_{t-1},x_0) \times q(x_{t-1}|x_0)}{q(x_t|x_0)} \\ & = \dfrac{q(x_t|x_{t-1}) \times q(x_{t-1}|x_0)}{q(x_t|x_0)} \end{aligned}$

其中，$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1},\beta_t\boldsymbol{I}) \Rightarrow x_t = \sqrt{1 - \beta_t}x_{t-1} + \sqrt{\beta_t}\varepsilon_{t-1}$，$q(x_{t-1}|x_0) = \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_0,(1 - \bar{\alpha}_{t-1})\boldsymbol{I})$，$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0,(1 - \bar{\alpha}_t)\boldsymbol{I})$

然后，通过正态分布的概率密度函数 $\mathcal{N}(\mu, \sigma) = \dfrac{1}{\sqrt{2\pi}\sigma}exp(-\dfrac{(x-\mu)^2}{2\sigma^2})$，对上式进行进一步化简可得：

$q(x_{t-1} | x_t,x_0) = \mathcal{N}(x_{t-1};\tilde{\mu}_t(x_t,x_0), \tilde{\beta}_t\boldsymbol{I})$，其中 $\tilde{\mu}_t(x_t,x_0) = \dfrac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}}x_t + \dfrac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}x_0,\ \tilde{\beta}_t = \dfrac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$

将 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\bar{\varepsilon}_0 \Rightarrow x_0 = \dfrac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\bar{\varepsilon}_0)$ 代入 $\tilde{\mu}_t(x_t,x_0)$ 可得：$\tilde{\mu}_t = \dfrac{1}{\sqrt{\alpha_t}}(x_t - \dfrac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\bar{\varepsilon}_0)$ 

此时，$q(x_{t-1}|x_t)$ 就只依赖 $x_t$ 和 $\bar{\varepsilon}_0$，即 $q(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \dfrac{1}{\sqrt{\alpha_t}}(x_t - \dfrac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\bar{\varepsilon}_0), \dfrac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t\boldsymbol{I})$ 

因此，我们只需要设计一个模型 $\varepsilon_{\theta}(x_t,t)$ 来通过输入 $x_t$ 和 $t$ 来预测添加的噪声 $\bar{\varepsilon}_0$，并使用 $MSE\ loss$ 计算损失：

$\begin{aligned} L_{\theta} & = E_{t \in [1,T],x_0,\bar{\varepsilon}_t}[||\bar{\varepsilon}_t - \varepsilon_\theta(x_t, t||^2] \\ & =  E_{t \in [1,T],x_0,\bar{\varepsilon}_t}[||\bar{\varepsilon}_t - \varepsilon_\theta( \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\bar{\varepsilon}_0, t||^2] \end{aligned}$

就可以实现模型训练。

在获得了 $\bar{\varepsilon}_0$ 后，想要获得 $x_{t-1}$，可以根据 $q(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \dfrac{1}{\sqrt{\alpha_t}}(x_t - \dfrac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\bar{\varepsilon}_0), \dfrac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t\boldsymbol{I})$ 得到 $x_{t-1} = \dfrac{1}{\sqrt{\alpha_t}}(x_t - \dfrac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\bar{\varepsilon}_0) + \dfrac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t \times z_t, z_t \in \mathcal{N}(0, \boldsymbol{I})$。所以，和直觉不同，再预测得到 $\bar{\varepsilon}_0$ 后，获得 $x_{t-1}$ 仍然需要一次随机采样，这就导致预测得到的 $\hat{x}_{t-1}$ 和原始的 $x_{t-1}$ 不完全一致，受 $z_t$ 的随机性影响。

此外，由于 $x_0 = \dfrac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}\bar{\varepsilon}_0)$，理论上也可以根据预测得到的 $\bar{\varepsilon}_0$，直接一步逆扩散到 $x_0$，但是没人这么做，说明效果很差，所以 DDPM 只在输入时使用一步扩散，而在预测时还是使用一步步的逆扩散。

DDIM
===

$\begin{align}x_{t-1} & = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\bar{\varepsilon}_{t-1}, \bar{\varepsilon}_{t-1} \sim \mathcal{N}(0, \boldsymbol{I}) \\ & = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}\bar{\varepsilon}_{t} + \sigma_t^2\bar{\varepsilon}, \sigma_t^2 = \eta \dfrac{\beta_t(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \\ & = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}\dfrac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}} + \sigma_t^2\bar{\varepsilon} \end{align}$

$q_\sigma(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1};\sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}\dfrac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma^2_tI)$

令 $\eta = 0$，$q_\sigma(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1};\sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\dfrac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}}, 0) \rightarrow$ **DDIM (1)**

令 $\{\tau_1,...,\tau_S\}, \tau_1 < ... < \tau_S \in [1, T], S < T$，$q_{\sigma, \tau}(x_{\tau_{i-1}}|x_{\tau_i},x_0) = \mathcal{N}(x_{\tau_{i-1}};\sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2}\dfrac{x_{\tau_i} - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma^2_tI) \rightarrow$ **Improved DDPM**

将两者结合，$q_{\sigma, \tau}(x_{\tau_{i-1}}|x_{\tau_i},x_0) = \mathcal{N}(x_{\tau_{i-1}};\sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\dfrac{x_{\tau_i} - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}}, 0) \rightarrow $ **DDIM**

$x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\dfrac{x_{\tau_i} - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}}$

$x_0$ is unknown, given a noisy observation $x_t$, first make a prediction of the corresponding $x_0$, and then use it to obtain a sample $x_{t-1}$ through the reverse conditional distribution $q_\sigma(x_{t-1}|x_t,x_0)$. The model  $\epsilon_\theta(x_t)$ attempts to predict $\epsilon_t$ from $x_t$, then according to $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}{\epsilon}_t$, $x_0 = \dfrac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}{\epsilon}_t)$. 

$\begin{align}x_{\tau_{i-1}} & = \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\dfrac{x_{\tau_i} - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1 - \bar{\alpha}_t}} \\ &  = \sqrt{\bar{\alpha}_{t-1}}\dfrac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1 - \bar{\alpha}_t}{\epsilon}_t) + \sqrt{1 - \bar{\alpha}_{t-1}}\dfrac{\textcolor{blue}{x_{\tau_i}} - {\color{Red}\sqrt{\bar{\alpha}_t}\dfrac{1}{\sqrt{\bar{\alpha}_t}}}({\color{Blue}x_t} - {\color{Orange}\sqrt{1 - \bar{\alpha}_t}}{\epsilon}_t)}{\color{Orange}\sqrt{1 - \bar{\alpha}_t}} \\ & = \sqrt{\bar{\alpha}_{t-1}}(\dfrac{x_t - \sqrt{1 - \bar{\alpha}_t}{\epsilon}_t}{\sqrt{\bar{\alpha}_t}}) + \sqrt{1 - \bar{\alpha}_{t-1}}\epsilon_t \end{align}$

Condition
===

## Classifier Guidance

train a classifier $f_\phi(y|x_t,t)$, and use gradients $\nabla_{x_t}log\ f_\phi(y|x_t)$ to guide the diffusion sampling process toward the conditioning information $y$. $\nabla_{x_t}log\ q(x_t) = - \dfrac{1}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta(x_t,t)$:

$\begin{align}\nabla_{x_t}log\ q(x_t,y) & = \nabla_{x_t}log\ q(x_t) + \nabla_{x_t}log\ q(y|x_t) \\ & \approx \nabla_{x_t}log\ q(x_t) + \nabla_{x_t}log\ f_\phi(y|x_t) \\ & = - \dfrac{1}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta(x_t,t) + \nabla_{x_t}log\ f_\phi(y|x_t) \\ & = - \dfrac{1}{\sqrt{1 - \bar{\alpha}_t}}(\epsilon_\theta(x_t,t) - \sqrt{1 - \bar{\alpha}_t}\nabla_{x_t}log\ f_\phi(y|x_t)) \end{align}$ 

a new classifier-guided predictor $\bar{\epsilon}_\theta(x,t) = \epsilon_\theta(x_t,t) - \sqrt{1 - \bar{\alpha}_t}\nabla_{x_t}log\ f_\phi(y|x_t)$. To control the strength of the classifier guidance, add a weight $\omega$ to the delta part: 

$\bar{\epsilon}_\theta(x,t) = \epsilon_\theta(x_t,t) - \sqrt{1 - \bar{\alpha}_t}\omega\nabla_{x_t}log\ f_\phi(y|x_t)$

DDPM: $\begin{align}x_{t-1} & = \dfrac{1}{\sqrt{\alpha_t}}(x_t - \dfrac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\bar{\epsilon}_\theta(x,t)) + \dfrac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t \times z_t, z_t \in \mathcal{N}(0, \boldsymbol{I}) \\ & = \dfrac{1}{\sqrt{\alpha_t}}(\underset{\mu_t}{\underbrace{x_t - \dfrac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta(x_t,t)}} + \sqrt{1 - \bar{\alpha}_t}\omega\nabla_{x_t}log\ f_\phi(y|x_t)) + \underset{\Sigma_t}{\underbrace{\dfrac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t}} \times z_t \end{align}$

DDIM: 

$\begin{align}x_{\tau_{i-1}} & \sqrt{\bar{\alpha}_{t-1}}(\dfrac{x_t - \sqrt{1 - \bar{\alpha}_t}\bar{\epsilon}_\theta(x,t)}{\sqrt{\bar{\alpha}_t}}) + \sqrt{1 - \bar{\alpha}_{t-1}}\bar{\epsilon}_\theta(x,t) \\ & = \sqrt{\bar{\alpha}_{\tau_{i-1}}}(\dfrac{x_{\tau_i} - \sqrt{1 - \bar{\alpha}_{\tau_i}}(\epsilon_\theta(x_t,t) - \sqrt{1 - \bar{\alpha}_t}\omega\nabla_{x_t}log\ f_\phi(y|x_t))}{\sqrt{\bar{\alpha}_{\tau_i}}}) + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}}}(\epsilon_\theta(x_t,t) - \sqrt{1 - \bar{\alpha}_t}\omega\nabla_{x_t}log\ f_\phi(y|x_t)) \end{align}$

The resulting *ablated diffusion model* (**ADM**) and the one with additional classifier guidance (**ADM-G**): 

![classifier_guidance](/images/classifier_guidance.png)

## Classifer-Free Guidance

Without an independent classifier $f_\phi(·)$,  unconditional denoising diffusion mode $p_\theta(x) \rightarrow \epsilon_\theta(x_t,t)$, conditonal model $p_\theta(x|y) \rightarrow \epsilon_\theta(x_t,t,y)$,  the conditioning information $y$ gets discarded periodically at random: $\epsilon_\theta(x_t,t) = \epsilon_\theta(x_t,t,y=\varnothing)$. 

$\nabla_{x_t}log\ p(y|x_t) = \nabla_{x_t}log\ p(x_t|y) - \nabla_{x_t}log\ p(x_t) = - \dfrac{1}{\sqrt{1 - \bar{\alpha}_t}}(\epsilon_\theta(x_t,t, y) - \epsilon_\theta(x_t,t)) \\ \begin{align}\bar{\epsilon}_\theta(x,t,y) & = \epsilon_\theta(x_t,t, y) - \sqrt{1 - \bar{\alpha}_t}\omega\nabla_{x_t}log\ p(y|x_t) \\ & = \epsilon_\theta(x_t,t, y) + \omega(\epsilon_\theta(x_t,t, y) - \epsilon_\theta(x_t,t)) \\ & = (\omega + 1)\epsilon_\theta(x_t,t, y) - \omega\epsilon_\theta(x_t,t) \end{align}$

附录
===

A. $q(x_{t-1}\vert x_t,x_0)$ 使用正态分布概率密度函数推导：

$$\begin{aligned}\mu_1 & = \sqrt{1 - \beta_t}x_{t-1};&\ \sigma_1^2 & = \beta_t &\\
\mu_2 & = \sqrt{\bar{\alpha}_{t-1}}x_0;&\ \sigma_2^2 & = 1 - \bar{\alpha}_{t-1} &\\
\mu_3 & = \sqrt{\bar{\alpha}_{t}}x_0;&\ \sigma_3^2 & = 1 - \bar{\alpha}_{t} &\end{aligned}$$

$$\begin{aligned} q(x_{t-1}\vert x_t,x_0) & \Rightarrow \dfrac{\dfrac{1}{\sqrt{2\pi}\sigma_1}exp(-\dfrac{(x-\mu_1)^2}{2\sigma_1^2}) \times \dfrac{1}{\sqrt{2\pi}\sigma_2}exp(-\dfrac{(x-\mu_2)^2}{2\sigma_2^2})}{\dfrac{1}{\sqrt{2\pi}\sigma_3}exp(-\dfrac{(x-\mu_3)^2}{2\sigma_3^2})} \\
& \Rightarrow \dfrac{1}{\sqrt{2\pi}\dfrac{\sigma_1\sigma_2}{\sigma_3}}exp(-\dfrac{(x-\mu_1)^2}{2\sigma_1^2}-\dfrac{(x-\mu_2)^2}{2\sigma_2^2}+\dfrac{(x-\mu_3)^2}{2\sigma_3^2}) \\
& \Rightarrow \sigma^2 = \dfrac{\sigma_1^2\sigma_2^2}{\sigma_3^2} = \dfrac{\beta_t(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \\
& \Rightarrow -\dfrac{(x-\mu)^2}{2\sigma^2}) = -\dfrac{(x-\mu_1)^2}{2\sigma_1^2}-\dfrac{(x-\mu_2)^2}{2\sigma_2^2}+\dfrac{(x-\mu_3)^2}{2\sigma_3^2} \\ & \Rightarrow \dfrac{x_t^2 - 2 \sqrt{\alpha_t}x_tx_{t-1}+\alpha_t\textcolor{red}{x_{t-1}^2} }{\beta_t} + \dfrac{\textcolor{red}{x_{t-1}^2} - 2\sqrt{\bar{\alpha}_{t-1}}x_0x_{t-1} + \bar{\alpha}_{t-1}x_0^2}{1 - \bar{\alpha}_{t-1}} - \dfrac{(x_t - \sqrt{\bar{\alpha}_t}x_0)^2}{1 - \bar{\alpha}_t} \\ & \Rightarrow \textcolor{red}{(\dfrac{\alpha_t}{\beta_t} + \dfrac{1}{1 - \bar{\alpha}_{t-1}})x_{t-1}^2 -} \textcolor{blue}{(\dfrac{2\sqrt{\alpha_t}}{\beta_t}x_t + \dfrac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}}x_0)x_{t-1}} + C(x_t,x_0) \\ & \Rightarrow \mu = \dfrac{2a}{b} = \dfrac{2(\dfrac{\alpha_t}{\beta_t} + \dfrac{1}{1 - \bar{\alpha}_{t-1}})}{\dfrac{2\sqrt{\alpha_t}}{\beta_t}x_t + \dfrac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}}x_0} \\ & \Rightarrow \mu = \dfrac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}}x_t + \dfrac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}x_0 \\ & \Rightarrow -\dfrac{(x-\dfrac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1 - \bar{\alpha}_{t}}x_t + \dfrac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}x_0)^2}{2(\dfrac{\beta_t(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t})^2}\end{aligned}$$

B. DDPM 代码框架：在训练时，首先，你需要预设置方差序列 $\{\beta_{t} \in (0, 1)\}_{t=1}^T$ 并计算 $\bar{\alpha}_{1:T}$。
然后，在 $1\sim T$ 中随机选择一个数字 $t$，并使用正态分布随机函数生成 $\bar{\varepsilon}_0$ (注意，这里生成的正态分布随机变量 $\bar{\varepsilon}_0$ 的维度为 $H \times W \times 3$，和原始图像 $x_0$ 一致)；
通过公式 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\bar{\varepsilon}_0$ 计算 $x_t$，其维度也为 $H \times \W \times 3$；
接着构造一个模型，输入 $x_t$ 和 $t$ (通常 $t$ 需要转化成 embedding，类似 Transformer，可以选择正弦函数这种确定的方式，也可以选择 learnable embedding parameter 让模型学习)，
输出和图像 $x_t$ 维度相同的噪声 $\varepsilon_\theta(x_t,t)$，因此一般选择 U-net 架构模型。
最后计算 $MSE$ 损失进行训练：$L_\theta = =  E_{t \in [1,T],x_0,\bar{\varepsilon}_0}[||\bar{\varepsilon}_0 - \varepsilon_\theta( \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\bar{\varepsilon}_0, t||^2]$。

而在推理时，首先使用正态分布随机函数生成 $\hat{x}^T$，维度为 $H \times W \times 3$，然后
将 $t=T$ 一起输入训练好的模型，预测输出 $\hat{\varepsilon}_0$，并使用正态分布随机函数生成 $z_t$，维度为 $H \times W \times 3$，
接着使用公式 $x_{t-1} = \dfrac{1}{\sqrt{\alpha_t}}(x_t - \dfrac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\bar{\varepsilon}_0) + \dfrac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t \times z_t$
生成预测的 $\hat{x_{T-1}}$，循环迭代，直到 $t = 0$ 时结束，最终的 $\hat{x}_0$ 即为模型生成的图像。

C. Classifier-Guidance 代码框架：由上述推导可知，最后需要将 classifier 的梯度加入到预测的噪声中：$\bar{\epsilon}_\theta(x,t) = \epsilon_\theta(x_t,t) - \sqrt{1 - \bar{\alpha}_t}\omega\nabla_{x_t}log\ f_\phi(y|x_t)$。注意，这里是 classifier 关于输入 $x_t$ 的梯度，而不是 classifier 模型参数的梯度。因此，我们可以利用 ```torch``` 的自动求导机制对 $x_t$ 进行求导，而由于 $x_t$ 的梯度和 $\epsilon_\theta(x_t,t)$ 形状相同(都是原始图像的形状)，因此我们可以直接将它们进行相加，具体代码框架如下：

```python
def cond_fn(x, t, y):
    """
    x 表示 x_t; y 表示 label
    """
    with th.enable_grad():
        x_in = x.detach().requires_grad_(True)  # 将 x 设置为需要梯度
        logits = classifier(x_in, t)  #  classifier 前向过程
        log_probs = F.log_softmax(logits, dim=-1)  # softmax 求概率
        selected = log_probs[range(len(logits)), y.view(-1)]
        # th.autograd.grad(selected.sum(), x_in) 是通过 selected 反向传播求解 x_in 的梯度，其形状和 x_in 一致；classifier_scale 即 $\omega$
        return th.autograd.grad(selected.sum(), x_in)[0] * classifier_scale
def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
    """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
    gradient = cond_fn(x, self._scale_timesteps(t), y)
    # p_mean_var["mean"] 表示原始的 $\epsilon$; p_mean_var["variance"] 表示 $\sqrt{1 - \bar{\alpha}_t}$
    new_mean = (p_mean_var["mean"] + p_mean_var["variance"] * gradient.float())
    return new_mean  # 经过 classifier-guidance 的 $\epsilon$
```

