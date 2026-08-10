---
title: "CodeContests-O: Powering LLMs via Feedback-Driven Iterative Test Case Generation"
collection: publications
# state: under review
permalink: /publication/2026-01-20-CodeContests-O
excerpt: "The rise of reasoning models necessitates large-scale verifiable data, for which programming tasks serve as an ideal source. To address this, we propose a Feedback-Driven Iterative Framework for comprehensive test case construction and release CodeContests-O."
date: 2026-01-20
venue: 'Association for Computational Linguistics'
paperurl: 'https://aclanthology.org/2026.findings-acl.53.pdf'

---
<p style="text-align:justify; text-justify:inter-ideograph;">The rise of reasoning models necessitates large-scale verifiable data, for which programming tasks serve as an ideal source. However, while competitive programming platforms provide abundant problems and solutions, high-quality test cases for verification remain scarce. Existing approaches attempt to synthesize test cases using Large Language Models (LLMs), but rely solely on the model's intrinsic generation capabilities without external feedback, frequently resulting in insufficiently diverse cases. To address this limitation, we propose a Feedback-Driven Iterative Framework for comprehensive test case construction. Specifically, our method leverages the LLM to generate initial test cases, executes them against known correct and incorrect solutions, and utilizes the failed results as feedback to guide the LLM in refining the test cases toward high fidelity and discriminability. We then apply this method to the CodeContests dataset to construct an optimized high-quality derivative, CodeContests-O. Evaluating against the entire pool of solutions (1.1 × 10<sup>7</sup> in total), our dataset achieves an average True Positive Rate (TPR) of 89.37% and True Negative Rate (TNR) of 90.89%, significantly outperforming the CodeContests and CodeContests+ by margins of 4.32% and 9.37%, respectively. Furthermore, fine-tuning the Qwen2.5-7B model on CodeContests-O results in a 9.52% improvement on LiveCodeBench (Pass@1). Experiments demonstrate the effectiveness of our framework and the quality of CodeContests-O. To support reproducibility and facilitate future research, we release the code and dataset.</p>

[Download paper from here](https://aclanthology.org/2026.findings-acl.53.pdf)

<p style="text-align:justify; text-justify:inter-ideograph;">BibTeX formatted citation: </p>

<pre>
@inproceedings{cai-etal-2026-codecontests,
    title = "{C}ode{C}ontests-{O}: Powering {LLM}s via Feedback-Driven Iterative Test Case Generation",
    author = "Cai, Jianfeng and Zhu, Jinhua and Sun, Ruopei and Zhao, Kangwen and Xue, Dongyun and Feng, Mingxiao and Zhou, Wengang and Li, Houqiang",
    editor = "Liakata, Maria and Moreira, Viviane P. and Zhang, Jiajun and Jurgens, David",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.53/",
    doi = "10.18653/v1/2026.findings-acl.53",
    pages = "1054--1072",
    ISBN = "979-8-89176-395-1"
}
</pre>
