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

在<a href="https://cai-jianfeng.github.io/posts/2024/04/blog-rlhf/" target="_blank">之前的博客</a>中，我们讲解了 RLHF 的三个阶段：SFT（预训练 LLM 模型 $M_\theta$），Reward Modeling （预训练奖励模型 $r_\theta$）和最后的 RL 训练（使用 PPO 微调 $M_\theta$）。对于前两个阶段而言，其只存在一个模型，因此可以使用 Deepspeed，FSDP，Megatron，甚至是 Transformers 的内置 Trainer 等<b>单模型</b>训练框架直接进行分布式训练（关于单模型训练框架，可以参考我<a href="https://cai-jianfeng.github.io/posts/2025/06/blog-distributed-train-pipeline/" target="_blank">之前的博客</a>）。对于第三个阶段而言，其包含多个模型，同时不同模型的所处状态也不尽相同（例如，reference model 和 reward model 只用于 infer，policy model 和 value model 用于 train，同时 policy model 还用于 rollout）。因此，需要在 Deepspeed/FSDP/Megatron 这种单模型的训练框架上再进行进一步的搭建以构建<b>多模型</b>训练框架。因此，本文的所有 RLHF 框架其实主要是聚焦于构建第三阶段的多模型训练框架。

下面，我们以 PPO 的为例来了解每个架构的整体架构和每个部分的具体模块（其他的 RL 算法如 GRPO，REINFORCE++ 等基本上都是在 PPO 的基础上减少某些模块）。首先，如下图所示，我们先逻辑化整理一下 PPO 的算法流程：


敬请期待🤪（争取端午节放假结束之前完成）