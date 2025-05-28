---
title: 'The Basic Knowledge of RLHF Training Pipeline'
date: 25-05-26
update: 25-05-26
permalink: /posts/2025/05/blog-rlhf-train-pipeline/
star: superior
tags:
  - 深度学习基本知识
---

<p style="text-align:justify; text-justify:inter-ideograph;">这篇博客主要讲解 RLHF 具体训练的框架（DeepSpeedChat，OpenRLHF，verl）的具体细节，包括每个框架的整体架构，架构内的各部分细节（包括逻辑细节和代码细节）。(建议先阅读我之前关于 RLHF 的博客 <a href="https://cai-jianfeng.github.io/posts/2024/04/blog-rlhf/" target="_blank">The Basic Knowledge of RLHF (Reinforce Learning with Human Feedback)</a>)</p>

RLHF 的算法流程
===

<p style="text-align:justify; text-justify:inter-ideograph;">在<a href="https://cai-jianfeng.github.io/posts/2024/04/blog-rlhf/" target="_blank">之前的博客</a>中，我们讲解了 RLHF 的三个阶段：SFT（预训练 LLM 模型 $M_\theta$），Reward Modeling （预训练奖励模型 $r_\theta$）和最后的 RL 训练（使用 PPO 微调 $M_\theta$）。对于前两个阶段而言，其只存在一个模型，因此可以使用 Deepspeed，FSDP，Megatron，甚至是 Transformers 的内置 Trainer 等<b>单模型</b>训练框架直接进行分布式训练（关于单模型训练框架，可以参考我<a href="https://cai-jianfeng.github.io/posts/2025/06/blog-distributed-train-pipeline/" target="_blank">之前的博客</a>）。对于第三个阶段而言，其包含多个模型，同时不同模型的所处状态也不尽相同（例如，reference model 和 reward model 只用于 infer，policy model 和 value model 用于 train，同时 policy model 还用于 rollout）。因此，需要在 Deepspeed/FSDP/Megatron 这种单模型的训练框架上再进行进一步的搭建以构建<b>多模型</b>训练框架。因此，本文的所有 RLHF 框架其实主要是聚焦于构建第三阶段的多模型训练框架。</p>

<p style="text-align:justify; text-justify:inter-ideograph;">下面，我们以 PPO 为例来了解每个框架的整体架构和每个部分的具体模块（其他的 RL 算法如 GRPO，REINFORCE++ 等基本上都是在 PPO 的基础上减少某些模块）。首先，如下图所示（这里直接盗用 <a href="https://arxiv.org/abs/2405.11143" target="_blank">OpenRLHF</a> 的图片🥳），我们先逻辑化整理一下 PPO 的算法流程：</p>

![ppo pipeline](/images/PPO_gen_and_learn.png)

<p style="text-align:justify; text-justify:inter-ideograph;"><b>A. PPO 的生成阶段：</b>即通过给定的输入，生成一系列 PPO 所训练的必要的元素，在经典 RL 中也被称作环境交互。</p>

1. <p style="text-align:justify; text-justify:inter-ideograph;">给定 SFT 后得到的 model，将其复制为 reference model $\pi_{SFT}$ 和需要进一步训练的 actor model $\pi_{RL}$；给定 Reward Modeling 后得到的 model，将其复制为 reward model $R$ 和 critic model $V$。</p>

2. <p style="text-align:justify; text-justify:inter-ideograph;">给定 prompt $x$，将其输入给 actor model $\pi_{RL}$ 生成对应的 response $y$，得到完整的 sequence $x + y$。（<span style="color: red;">$\pi_{RL}$ rollout</span>）</p>

3. <p style="text-align:justify; text-justify:inter-ideograph;">给定 sequence $x + y$，将其输入给 actor model $\pi_{RL}$ 和 reference model $\pi_{SFT}$ 分别生成 action logits $p_{RL}$ 和 sft logits $p_{SFT}$，并进一步计算 KL divergence $KL$。（<span style="color: red;">$\pi_{RL}$ 和 $\pi_{SFT}$ infer</span>）</p>

4. <p style="text-align:justify; text-justify:inter-ideograph;">给定 sequence $x + y$，将其输入给 reward model $R$ 和 critic model $V$ 分别生成 reward $r$ 和 value $v$。（<span style="color: red;">$R$ 和 $V$ infer</span>）</p>

5. <p style="text-align:justify; text-justify:inter-ideograph;">给定 $KL$ 和 $r$，计算得到 PPO 的 return；并通过给定 $v$，计算得到 PPO 的 advantage $A$。</p>

<p style="text-align:justify; text-justify:inter-ideograph;"><b>B. PPO 的训练阶段：</b>即通过 PPO 的生成阶段所得到的元素，进行 PPO 的训练，在经典 RL 中也被称作奖励学习。由于 PPO 的生成阶段的时间成本较高，因此通常对生成阶段得到的元素进行缓存，并进行多次训练。对于第 $t$ 次训练，其具体流程如下：</p>

1. <p style="text-align:justify; text-justify:inter-ideograph;">给定 sequence $x + y$，将其输入给第 $t-1$ 次训练完的 actor model $\pi_{RL}$ 生成 new action logits $p^{t}_{RL}$，并和 action logits $p_{RL}$ 计算 ratio $r^{t}(\theta)$。接着和给定的 advantage $A$ 计算 Actor loss 用于 actor model 的训练。</p>

2. <p style="text-align:justify; text-justify:inter-ideograph;">给定 sequence $x + y$，将其输入给第 $t-1$ 次训练完的 critic model $V$ 生成 new value $v^{t}$，并和 value $v$ 计算 clipped value $v_{clip}$。接着和给定的 return 计算 Critic loss 用于 critic model 的训练。</p> 

DeepSpeedChat
===

可以看到，上述的流程涉及到 actor model $\pi_{RL}$ 的 rollout，actor model $\pi_{RL}$ 和 $\pi_{SFT}$ 的 infer，reward model $R$ 和 critic model $V$ 的 infer，以及 actor model $\pi_{RL}$ 和 critic model $V$ 的 train。<b>最直接的实现方式是，按照上述流程的逻辑编写 PPO 训练的架构，通过简单扩展单模型训练框架得到多模型训练框架。</b>DeepSpeedChat 就是按照这种思路扩展 DeepSpeed 框架来实现 PPO 的训练的。



敬请期待🤪（争取端午节放假结束之前完成）