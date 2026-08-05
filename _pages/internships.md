---
layout: archive
title: "Internships"
permalink: /internships/
author_profile: true
excerpt: "Research and engineering internships spanning LLM post-training, agents, RAG, and 3D vision."
---

{% include base_path %}

<p class="internship-list__intro">Research and engineering internships spanning LLM post-training, agents, retrieval-augmented generation, and 3D vision.</p>

{% assign internships = site.internships | sort: "start_date" | reverse %}

<div class="internship-list" role="list" aria-label="Internship history">
  {% for internship in internships %}
    {% include internship-card.html internship=internship %}
  {% endfor %}
</div>
