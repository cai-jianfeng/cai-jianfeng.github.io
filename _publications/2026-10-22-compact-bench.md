---
title: "Compact-Bench: Measuring What Context Compaction Costs Long-Horizon Coding Agents"
collection: publications
state: under review
permalink: /publication/2026-10-22-compact-bench
excerpt: 'Compact-Bench isolates the cost of context compaction for long-horizon coding agents by comparing full-history and summary-only continuations from shared repository states across task preservation, recurring-detail retention, and production-chain retention.'
date: 2026-10-22
venue: 'arXiv'
paperurl: 'https://arxiv.org/abs/2610.31710'
---

Ruopei Sun<sup>*</sup>, **Jianfeng Cai**<sup>*</sup>, Kangwen Zhao, Wengang Zhou, and Houqiang Li.

<small><sup>*</sup> Equal contribution.</small>

<p style="text-align:justify; text-justify:inter-ideograph;">Context compaction allows long-horizon coding agents to continue beyond their context windows, but replacing the full interaction history with a summary can remove task state required for successful completion. Compact-Bench isolates this cost by pairing full-history and summary-only continuations from the same repository state and evaluating three complementary dimensions: task preservation, recurring-detail retention, and retention across successive production compactions. The benchmark includes 35 SWE-bench repairs, 146 agentic sessions, and 106 production chains.</p>

<p style="text-align:justify; text-justify:inter-ideograph;">Across 14 summarizer backbones, stable damage ranges from 5% to 64% of the 22 repairs solved with full histories. With Kimi-K3 as the summarizer, the Kimi Code prompt preserves the most task continuations (19/22), while Claude Code retains the most recurring details (15.0%). LLM summaries retain at most 16.0% of recurring details, compared with 79.1% for mechanical extraction. These findings show that reliable compaction must preserve the progress needed to finish a task rather than merely retain information that appears relevant.</p>

[Download paper from here](https://arxiv.org/pdf/2610.31710) or [view the arXiv page](https://arxiv.org/abs/2610.31710). The [code](https://github.com/cai-jianfeng/compact-bench) and [dataset](https://huggingface.co/datasets/caijanfeng/compact-bench) are publicly available.

<p style="text-align:justify; text-justify:inter-ideograph;">BibTeX formatted citation: </p>

<pre>
@misc{sun2026compactbenchmeasuringcontextcompaction,
      title={Compact-Bench: Measuring What Context Compaction Costs Long-Horizon Coding Agents},
      author={Ruopei Sun and Jianfeng Cai and Kangwen Zhao and Wengang Zhou and Houqiang Li},
      year={2026},
      eprint={2610.31710},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2610.31710}
}
</pre>
