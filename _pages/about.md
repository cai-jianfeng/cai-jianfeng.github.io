---
permalink: /
title: <center>Cai-jianfeng Academic Personal Websites</center>
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% include base_path %}

This is the front page of the cai-jianfeng academic personal websites.

Studying
======
  <ul>{% for post in site.teaching %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

Skills
======
* Programming Capability
* Innovation and Creativity
* Team-Work Ability

Publications
======
  <ul>{% for post in site.publications %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

Competitions
======
  <ul>{% for post in site.talks %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
