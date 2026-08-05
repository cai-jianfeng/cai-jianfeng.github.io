---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% include base_path %}

<p class="archive-intro">Research publications and technical reports, listed in reverse chronological order.</p>

{% assign publications = site.publications | sort: "date" | reverse %}
<div class="archive-list">
  {% for post in publications %}
    {% include archive-single.html %}
  {% endfor %}
</div>
