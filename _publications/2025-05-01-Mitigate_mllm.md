---
title: "Mitigating Hallucination in VideoLLMs via Temporal-Aware Activation Engineering"
collection: publications
# state: under review
permalink: /publication/2025-05-01-mitigate-mllm
excerpt: 'It is the first to systematically investigate the effectiveness and underlying mechanisms of activation engineering for mitigating hallucinations in VideoLLMs. And it proposes a temporal-aware activation engineering framework for VideoLLMs, which adaptively identifies and manipulates hallucination-sensitive modules based on the temporal variation characteristic, substantially mitigating hallucinations without additional LLM fine-tuning.'
date: 2025-05-01
venue: 'Neural Information Processing Systems'
paperurl: 'https://openreview.net/forum?id=7mTECPRtll'
# citation: 'Cai Jianfeng. (2025). &quot;Mitigating Hallucination in VideoLLMs via Temporal-Aware Activation Engineering.&quot; <i>arXiv preprint arXiv: 2505.12826, 2025</i>.'
---
<p style="text-align:justify; text-justify:inter-ideograph;">Multimodal large language models (MLLMs) have achieved remarkable progress in video this http URL, hallucination, where the model generates plausible yet incorrect outputs, persists as a significant and under-addressed challenge in the video domain. Among existing solutions, activation engineering has proven successful in mitigating hallucinations in LLMs and ImageLLMs, yet its applicability to VideoLLMs remains largely unexplored. In this work, we are the first to systematically investigate the effectiveness and underlying mechanisms of activation engineering for mitigating hallucinations in VideoLLMs. We initially conduct an investigation of the key factors affecting the performance of activation engineering and find that a model's sensitivity to hallucination depends on **temporal variation** rather than task type. Moreover, selecting appropriate internal modules and dataset for activation engineering is critical for reducing hallucination. Guided by these findings, we propose a temporal-aware activation engineering framework for VideoLLMs, which adaptively identifies and manipulates hallucination-sensitive modules based on the temporal variation characteristic, substantially mitigating hallucinations without additional LLM fine-tuning. Experiments across multiple models and benchmarks demonstrate that our method markedly reduces hallucination in VideoLLMs, thereby validating the robustness of our findings.</p>

[Download paper from here](https://openreview.net/forum?id=7mTECPRtll)

<!-- <p style="text-align:justify; text-justify:inter-ideograph;">Cai Jianfeng. (2025). &quot;Mitigating Hallucination in VideoLLMs via Temporal-Aware Activation Engineering.&quot; <i>arXiv preprint arXiv: 2505.12826, 2025</i>.</p> -->

<p style="text-align:justify; text-justify:inter-ideograph;">BibTeX formatted citation: </p>

<pre>
@inproceedings{CaiHZZZL25,
  author       = {Jianfeng Cai and Jiale Hong and Zongmeng Zhang and Wengang Zhou and Nianji Zhan and Houqiang Li},
  editor       = {Danielle Belgrave and Cheng Zhang and Laura N. Montoya and Hsuan{-}Tien Lin and Razvan Pascanu and Piotr Koniusz and Marzyeh Ghassemi and Nancy Chen and Iv{\'{a}}n Vladimir Meza Ru{\'{\i}}z and Arturo Loaiza{-}Bonilla},
  title        = {Mitigating Hallucination in VideoLLMs via Temporal-Aware Activation Engineering},
  booktitle    = {Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025},
  year         = {2025},
  url          = {http://papers.nips.cc/paper_files/paper/2025/hash/5683583b88da51c79d2eb263f295286e-Abstract-Conference.html}
}
</pre>