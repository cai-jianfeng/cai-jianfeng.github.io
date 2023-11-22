### Score function

**score functions** (gradients of log probability density functions)

learn score functions on a large number of noise-perturbed data distributions, then generate samples with Langevin-type sampling.

denote data distribution is $\boldsymbol{p(x)}$, the goal of generative modeling is to fit a model to the data distribution such that we can synthesize new data points at will by sampling from the distribution.

assume probability density function: $\boldsymbol{p_\theta(x) = \dfrac{e^{-f_\theta(x)}}{Z_\theta},Z_\theta > 0, \int p_\theta(x)dx = 1}$, it is unnormalized probabilistic model, or energy-based model: $\boldsymbol{f_\theta(x)}$, 

train $\boldsymbol{p_\theta(x)}$ by maximizing the log-likelihood of the data: $\underset{\theta}{max}\sum_{i=1}^{N}{log\ \boldsymbol{p_\theta(x_i)}}$, requires $\boldsymbol{p_\theta(x)}$ to be a normalized probability density function $\Rightarrow$ evaluate the normalizing constant $Z_\theta\ \Rightarrow$ a typically intractable quantity for any general $\boldsymbol{f_\theta(x)}$

The **score function** of a distribution $\boldsymbol{p(x) = \triangledown_xlog\ p(x)}$, a model for the score function is called a **score-based model** $\boldsymbol{s_\theta(x) \approx \triangledown_xlog\ p(x)}$, it can be parameterized without worrying about the normalizing constant $\boldsymbol{Z_\theta} \Leftrightarrow$ the score-based model $\boldsymbol{s_\theta(x)}$ is independent of the normalizing constant $\boldsymbol{Z_\theta}$.

$\boldsymbol{s_\theta(x) \approx \triangledown_xlog\ p_\theta(x) = \triangledown_xlog \dfrac{e^{-f_\theta(x)}}{Z_\theta} = -\triangledown_xf_\theta(x) -\underset{=0}{\underbrace{ \triangledown_xlog\ Z_\theta}}} = -\triangledown_xf_\theta(x)$

train score-based models by minimizing the **Fisher divergence** between the model and the data distributions: $\mathbb{E}_{p(x)}[||\triangledown_xlog\ p(x) - s_\theta(x)||_2^2]$

unknown data score $\triangledown_xlog\ p(x)\ \Rightarrow$ **score matching**

Score matching objectives can directly be estimated on a dataset and optimized with stochastic gradient descent.

$1$-$D$ data mathematical derivation: 

$\begin{align}\mathbb{E}_{p(x)}[||\triangledown_xlog\ p(x) - s_\theta(x)||_2^2]  & = 2 * \dfrac{1}{2} \mathbb{E}_{p_{data}(x)}[||\triangledown_xlog\ p_{data}(x) - \triangledown_xlog\ p_\theta(x)||_2^2] \\ &  = 2 * \dfrac{1}{2} \mathbb{E}_{p_{data}(x)}[(\triangledown_xlog\ p_{data}(x) - \triangledown_xlog\ p_\theta(x))^2] \\ & = 2 * \dfrac{1}{2} \int p_{data}(x)(\triangledown_xlog\ p_{data}(x) - \triangledown_xlog\ p_\theta(x))^2 dx \\ & = 2  *(\underset{const}{\underbrace{\int \dfrac{1}{2} p_{data}(x)(\triangledown_xlog\ p_{data}(x))^2 dx}} + \int \dfrac{1}{2} p_{data}(x)(\triangledown_xlog\ p_{\theta}(x))^2 dx - \int p_{data}(x)\triangledown_xlog\ p_{\theta}(x)\triangledown_xlog\ p_{data}(x) dx) \end{align}$

$\int \dfrac{1}{2} p_{data}(x)(\triangledown_xlog\ p_{\theta}(x))^2 dx = \dfrac{1}{2} \mathbb{E}_{p_{data}}[(\triangledown_xlog\ p_{\theta}(x))^2]$

By integration by parts:

 $\begin{align} & - \int p_{data}(x)\triangledown_xlog\ p_{\theta}(x)\triangledown_xlog\ p_{data}(x) dx \\ = & - \int \triangledown_xlog\ p_{\theta}(x)\triangledown_x\ p_{data}(x) dx \\ = & -p_{data}(x)\triangledown_xlog\ p_{\theta}(x)|^\infty_{-\infty} + \int p_{data}(x)\triangledown_x^2log\ p_{\theta}(x)dx \\ \overset{(i)}{=} &\ \mathbb{E}_{p_{data}}[\triangledown_x^2log\ p_{\theta}(x)] \Leftarrow |x| \rightarrow 0, p_{data}(x) \rightarrow 0 \end{align}$

$\begin{align}\mathbb{E}_{p(x)}[||\triangledown_xlog\ p(x) - s_\theta(x)||_2^2]  = 2\ \mathbb{E}_{p_{data}}[\triangledown_x^2log\ p_{\theta}(x)] + \mathbb{E}_{p_{data}}[(\triangledown_xlog\ p_{\theta}(x))^2] + const\end{align}$

muti-dimensional data **score matching objective**, $\triangledown_x^2$ denotes the Hessian with respect to $x$: 

$\begin{align}\mathbb{E}_{p(x)}[||\triangledown_xlog\ p(x) - s_\theta(x)||_2^2] = 2\ \mathbb{E}_{p_{data}}[tr(\triangledown_x^2log\ p_{\theta}(x)) + \dfrac{1}{2}||\triangledown_xlog\ p_{\theta}(x)||_2^2] +const \end{align}$

compute $||\triangledown_xlog\ p_{\theta}(x)||_2^2 = ||\triangledown_xf_{\theta}(x)||_2^2 = ||s_{\theta}(x)||_2^2$,  $tr(\triangledown_x^2log\ p_{\theta}(x)) = tr(\triangledown_x^2f_{\theta}(x)) = tr(\triangledown_xs_{\theta}(x))$

if $f_\theta(x)$ is parameterized by a deep neural network, $tr(\triangledown_x^2log\ p_{\theta}(x)) $ requires a number of backpropagation that is proportional to the data dimension $D$. 

***sliced score matching***: one dimensional data distribution is much easier to estimate for score matching. project the scores onto random directions, such that the vector fields of scores of the data and model distribution become scalar fields. then compare the scalar fields to determine how far the model distribution is from the data distribution. It is clear to see that the two vector fields are equivalent if and only if their scalar fields corresponding to projections onto all directions are the same.

denote $\boldsymbol{v}$ as the random projection direction, The random projected version of Fisher divergence is ***sliced Fisher divergence***: 

$2 * \dfrac{1}{2}\mathbb{E}_{p_{data}}[(v^T\triangledown_xlog\ p_{data}(x) - v^T\triangledown_xlog\ p_{\theta}(x))^2] = 2 * (\mathbb{E}_{p_{data}}[v^T\triangledown_x^2log\ p_{\theta}(x) - \dfrac{1}{2}(v^T\triangledown_xlog\ p_{\theta}(x))^2] + const)$

$v^T\triangledown_x^2log\ p_{\theta}(x)$ is in the form of Hessian-vector products, which can be computed within $O(1)$ backpropagations

### Langevin dynamics

given trained $\boldsymbol{s_\theta(x) \approx \triangledown_xlog\ p(x)}$, we can use an iterative procedure called **Langevin dynamics** to draw samples from it.

Langevin dynamics provides an MCMC procedure to sample from a distribution $\boldsymbol{p(x)}$ using only its score function $\triangledown_xlog\ p(x)$, Specifically, it initializes the chain from an arbitrary prior distribution $x_0 \sim \pi(x)$, and then iterates the following:

$x_{i+1} \leftarrow x_i + \epsilon \triangledown_xlog\ p(x) + \sqrt{2\epsilon} z_i, i = 0,1,...,K, z_i \sim \mathcal{N}(0,I) \\ x_{i+1} \leftarrow x_i + \epsilon s_\theta(x) + \sqrt{2\epsilon} z_i$

When $ \rightarrow 0$ and $K \rightarrow \infty$, $x_K$ converges to a sample from $p(x)$.

### Pitfalls

The key challenge is the fact that the estimated score functions are inaccurate in low density regions, where few data points are available for computing the score matching objective. This is expected as score matching minimizes the Fisher divergence.

$\mathbb{E}_{p(x)}[||\triangledown_xlog\ p(x) - s_\theta(x)||_2^2] = \int p(x) ||\triangledown_xlog\ p(x) - s_\theta(x)||_2^2 dx$

score-based model are weighted by $p(x)$, they are largely ignored in low density regions where $p(x)$ is small.

### Multiple noise perturbations

**perturb** data points with noise and train score-based models on the noisy data points instead. When the noise magnitude is sufficiently large, it can populate low data density regions to improve the accuracy of estimated scores.

choose an appropriate noise scale for the perturbation process: use multiple scales of noise perturbations simultaneously. Suppose we always perturb the data with isotropic Gaussian noise, and let there be a total of $L$ increasing standard deviations $\sigma_1 < ... < \sigma_L$. 

1. perturb the data distribution $p(x)$ with each of the Gaussian noise $\mathcal{N}(0, \sigma^2_iI), i=1,...,L$ to obtain a noise-perturbed distribution: $p_{\sigma_i}(x) = \int p(y)\mathcal{N}(x;y,\sigma_i^2I)dy \rightarrow$ sample $x \sim p(x)$ and $z \sim \mathcal{N}(0,I)$ and compute $x + \sigma_iz$.

2. estimate the score function of each noise-perturbed distribution, $\triangledown_xlog\ p_{\sigma_i}(x)$, by training a **Noise Conditional Score-Based Model** $s_\theta(x,i)$ with score matching, such that $s_\theta(x,i) \approx \triangledown_xlog\ p_{\sigma_i}(x), i=1,...,L$.

   $\begin{align}\mathbb{E}_{p_{\sigma_i}(x)}[||\triangledown_xlog\ p_{\sigma_i}(x) - s_\theta(x, i)||_2^2] & = 2\ \mathbb{E}_{p_{data, \sigma_i}}[tr(\triangledown_x^2log\ p_{\theta}(x, i)) + \dfrac{1}{2}||\triangledown_xlog\ p_{\theta}(x, i)||_2^2] +const \\ & = 2\ \mathbb{E}_{p_{data, \sigma_i}}[tr(\triangledown_xs_{\theta}(x, i)) + \dfrac{1}{2}||s_{\theta}(x, i)||_2^2] + const \end{align}$

3. The training objective for $s_\theta(x,i)$ is a weighted sum of Fisher divergences for all noise scales: $\mathbb{E}_{p(x)}[||\triangledown_xlog\ p(x) - s_\theta(x,i)||_2^2] = \sum_{i=1}^L\lambda(i)\mathbb{E}_{p_{\sigma_i}(x)}[||\triangledown_xlog\ p_{\sigma_i}(x) - s_\theta(x, i)||_2^2], \lambda(i) \in \mathbb{R}_{>0}, \lambda(i) = \sigma_i^2$ 

4. **annealed Langevin dynamics**: produce samples from it by running Langevin dynamics for $i = L,L-1,...,1$ in sequence, since the noise scale $\sigma_i$ decreases (anneals) gradually over time.

   ![1700052639467](C:\Users\86199\AppData\Roaming\Typora\typora-user-images\1700052639467.png)

### Stochastic differential equations (SDEs)

By generalizing the number of noise scales to infinity, we obtain not only **higher quality samples**, but also, among others, **exact log-likelihood computation**, and **controllable generation for inverse problem solving**.

When the number of noise scales approaches infinity, we essentially perturb the data distribution with continuously growing levels of noise. In this case, the noise perturbation procedure is a continuous-time stochastic process.

Many stochastic processes (diffusion processes in particular) are solutions of stochastic differential equations (SDEs, **hand designed**):

$dx = f(x,t)dt + g(t)dw, f(·，t):\mathbb{R}^d \rightarrow \mathbb{R}^d, g(t) \in \mathbb{R} \Rightarrow dx = e^tdw$

$f(·，t)$ called **drift coefficient**, $g(t)$ called **diffusion coefficient**, $w$ denotes a standard Brownian motion. The solution of a stochastic differential equation is a continuous collection of random variables $\{x(t)\}_{t \in [0,T]}$. These random variables trace stochastic trajectories as the time index $t$ grows from the start time $0$ to the end time $T$. $p_t(x)$ denote the (marginal) probability density function of $x(t),\ p_0(x) = p(x), p_T(x) \approx \pi(x)$. 

$\boldsymbol{dx = e^tdw}$: perturbs data with a Gaussian noise of mean zero and exponentially growing variance ($f(x,t) = 0, g(t) = e^t$).

reverse the perturbation process for sample generation by using the reverse SDE. any SDE has a corresponding reverse SDE:

$dx = [f(x,t) - g^2(t)\triangledown_xlog\ p_t(x)]dt + g(t)dw$

train a **Time-Dependent Score-Based Model** $s_\theta(x,t) \approx \triangledown_xlog\ p_t(x)$:

$\begin{align} & \mathbb{E}_{t \sim \mathcal{U}(0,T)}\mathbb{E}_{p_t(x)}[\lambda(t)||\triangledown_xlog\ p_t(x) - s_\theta(x,t)||_2^2], \lambda: \mathbb{R} \rightarrow \mathbb{R}_{>0}, \lambda(t) \propto \dfrac{1}{\mathbb{E}[||\triangle_{x(t)log\ p(x(t))|x(0)}||_2^2]} \\ = & \mathbb{E}_{t \sim \mathcal{U}(0,T)} 2\ \mathbb{E}_{p_{t}(x)}[tr(\triangledown_x^2log\ p_t(x)) + \dfrac{1}{2}||\triangledown_xlog\ p_t(x)||_2^2] +const \\ = & 2\ \mathbb{E}_{t \sim \mathcal{U}(0,T)}\mathbb{E}_{p_{t}(x)}[tr(\triangledown_xs_{\theta}(x, t)) + \dfrac{1}{2}||s_{\theta}(x, t)||_2^2] + const \end{align}$

$dx = [f(x,t) - g^2(t)s_\theta(x,t)]dt + g(t)dw$

start with $x(T) \sim \pi$, and solve the above reverse SDE to obtain a sample $x(0), x(0) \sim p_\theta(x)$. 

**likelihood weighting function**: $\lambda(t) = g^2(t) \Rightarrow KL(p_0(x)||p_\theta(x)) \leq \dfrac{T}{2} \mathbb{E}_{t \sim \mathcal{U}(0,T)}\mathbb{E}_{p_t(x)}[\lambda(t)||\triangledown_xlog\ p_t(x) - s_\theta(x,t)||_2^2] + KL(p_T|| \pi)$

solving the estimated reverse SDE with numerical SDE solvers(Euler-Maruyama method), it discretizes the SDE using finite time steps and small Gaussian noise. 

$\triangle x \rightarrow [f(x,t) - g^2(t)s_\theta(x,t)]\triangle t + g(t) \sqrt{|\triangle t|}z_t, x \rightarrow x + \triangle x, t \rightarrow t + \triangle t, \triangle t < 0, \triangle t \approx 0, z_t \sim \mathcal{N}(0,I)$

### Probability flow ODE

a sampler based on ordinary differential equations (ODEs) that allow for exact likelihood computation. 

it s possible to convert any SDE into an ordinary differential equation (ODE) without changing its marginal distributions $\{p_t(x\}_{t \in [0,T]}$. The corresponding ODE of an SDE is named **probability flow ODE**: $dx = [f(x,t) - \dfrac{1}{2}g^2(t)\triangledown_xlog\ p_t(x)]dt$.

trajectories obtained by solving the probability flow ODE have the same marginal distributions as the SDE trajectories. can leverage the instantaneous change-of-variable formula to compute the unknown data density $p_0$ from the known prior density $p_T$ with numerical ODE solvers.

$$\underset{solutions}{\underbrace{\left[ \begin{array}{c} z_0 \\ log\ p(x) - log\ p_{z_0}(z_0) \end{array}\right]}} = \left[ \begin{array}{c} x \\ 0 \end{array}\right] + \underset{dynamics}{\underbrace{\int_{t_1}^{t_0} \left[ \begin{array}{c} f(z(t),t;\theta), Tr(\dfrac{\partial f}{\partial z(t)}) \end{array}\right]}} dt \\ \underset{inital\ values}{\underbrace{\left[ \begin{array}{c} z(t_1) \\ log\ p(x)-log\ p(z(t_1)) \end{array}\right] = \left[ \begin{array}{c} x \\ 0 \end{array}\right]}}$$

### Controllable generation

At its core, inverse problems are same as Bayesian inference problems. $x$ and $y$ are two random variables, $p(y|x)$ is know (the foreward process of generating $y$ from $x$). From Baye's rule:

$p(x|y) = p(x)p(y | x) / \int p(x)p(y|x)dx \Rightarrow \triangledown_xlog\ p(x|y) = \triangledown_xlog\ p(x) + \triangledown_xlog\ p(y|x), s_\theta(x) \approx \triangledown_xlog\ p(x|y)$

sample from $x$ using $\triangledown_xlog\ p(x|y)$ with Langevin-type sampling. For example, Class-conditional generation with an unconditional time-dependent score-based model $s_\theta(x)$, and a pre-trained noise-conditional image classifier $p(y|x)$, $y$ is class label, $x$ is image.

### Connection with Diffusion Model

By generalizing the number of noise scales to infinity, we proved that score-based generative models and diffusion probabilistic models can both be viewed as discretizations to stochastic differential equations determined by score functions, bridges both score-based generative modeling and diffusion probabilistic modeling into a unified framework. 
