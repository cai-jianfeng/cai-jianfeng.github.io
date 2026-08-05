---
title: 'The Basic Knowledge of Scaling Laws'
date: 24-05-29
update: 24-05-29
permalink: /posts/2024/05/blog-scaling-laws/
star: superior
tags:
  - 计算机基本知识
---


1. 对于一个固定的 batch size $B=512 \times 1024$：

|   Parameters   |   Data   |   Compute   |   Batch Size   | Equation |
| :--: | :--: | :--: | :--: | :--: |
| $N$ | $\infty$ | $\infty$ | Fixed | $L(N)=(N_c/N)^{\alpha_N}$ |
| $\infty$ | $D$ | Early Stop | Fixed | $L(D)=(D_c/D)^{\alpha_D}$ |
| Optimal | $\infty$ | $C$ | Fixed | $L(C)=(C_c/C)^{\alpha_C}$ |

2. $$B_{crit}(L) = \frac{B_*}{L^{1 / \alpha_B}}, B_* \sim 2 \cdot{10^8} tokens, \alpha_B \sim{0.21}$$：最优的 batch size 与 $N$ 无关，与 $L$ 成负相关，即 $L$ 越小，最优的 batch size 越大。

3. $$N\propto C_{\min}^{0.73}, B\propto C_{\min}^{0.24}, \mathrm{and}\ S\propto C_{\min}^{0.03}$$：增加算力时，应该大幅增加 model 大小，而后是 batch size 大小，最后是 step 大小。

4. $$D \propto N^{\frac{\alpha_N}{\alpha_D}} \sim N^{0.74}$$：model 每增加 $1$ 倍；data 需要增加 $0.74$ 倍。

5. $$L(N, D) = [(\frac{N_c}{N})^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D}]^{\alpha_D}, D\gtrsim(5\times10^3)N^{0.74}$$

6. $$L(N,S) = (\frac{N_c}{N})^{\alpha_N} + (\frac{S_c}{S_{min}(S)})^{\alpha_S}, S_c \approx 2.1 \times{10^3}, \alpha_S \approx 0.76; \\ \left(\frac{S}{S_{\min}}-1\right)\left(\frac{E}{E_{\min}}-1\right)=1, E = BS; \\ S_{\min}(S)\equiv\frac{S}{1+B_{\text{crit}}(L)/B}\quad\text{(minimum steps, at}\ B\gg B_{\text{crit}}); \\ C_{\min}(C)\equiv\frac{C}{1+B/B_{\text{crit}}(L)}\quad(\text{minimum compute, at}\ B\ll B_{\text{crit}}), C=6NBS$$：This means that in the model of size $N$, if using steps $S$ with batch size $B$ can reach test loss $L$, then the minimum number of steps in the model of size $N$ is $S_{min}(S)$ with batch size $B \propto \infty$ to reach test loss $L$.

7. $$S_{\mathrm{stop}}(N,D)\gtrsim\frac{S_c}{\left[L(N,D)-L(N,\infty)\right]^{1/\alpha_S}}$$

Performance penalty: $N^{0.74} / D \rightarrow$ increase $8\times$ model size, increase $5\times$ data
Training curves $\rightarrow$ power-law (independent of model size $N$)
Transfer to a different distribution incurs a constant penalty, and improves in line with performance on the training set
Large models are more sample-efficient than small models, reaching the same level of performance with fewer optimization steps using fewer data points
Training very large models and stopping significantly short of convergence (Compute-efficient)
The ideal batch size is the power of the loss only, and is determinable by measuring the gradient noise scale
Generalization depends almost exclusively on the in-distribution validation loss, and does not depend on the duration of training or proximity to convergence


Reference
===

1. [Scaling Laws for Neural Language Models](https://medium.com/@checkpoint89/scaling-laws-for-neural-language-models-fa1c0790833d)

