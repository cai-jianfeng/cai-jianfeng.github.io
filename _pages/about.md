---
permalink: /
title: <center><a href="https://cai-jianfeng.github.io/" title="Jianfeng Cai">Jianfeng Cai (蔡建峰)</a></center>
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% include base_path %}

<p style="text-align:justify; text-justify:inter-ideograph;">Master Student, supervised by Prof. <a href="http://staff.ustc.edu.cn/~lihq/" title="Houqiang Li" target="_blank">Houqiang Li</a> and <a href="http://staff.ustc.edu.cn/~zhwg/index.html" title="Wengang Zhou" target="_blank">Wengang Zhou</a>, Microsoft Key Laboratory of Multimedia Computing and Communication Lab, <a href="https://eeis.ustc.edu.cn/main.htm" title="eeis-ustc" target="_blank">Department of Electronic Engineering and Information Science, University of Science and Technology of China</a></p>

<p style="text-align:justify; text-justify:inter-ideograph;"><a href="https://cai-jianfeng.github.io/publications/" title="publications">Publications</a> | <a href="https://cai-jianfeng.github.io/talks/" title="competitions">Competitions</a> | <a href="https://cai-jianfeng.github.io/teaching/" title="studying">Studying</a> | <a href="https://cai-jianfeng.github.io/year-archive/" title="blog">Blog Posts</a></p>

<p style="text-align:justify; text-justify:inter-ideograph;">If you have any things or questions to discuss, do not hesitate to contact me.</p>

[[Wechat](/images/wechat.jpg)] [[QQ](/images/qq.jpg)] [<a href="mailto:xiaobaicai@mail.ustc.edu.cn" title="publications">school email</a>] [<a href="mailto:cjf1622613693@gmail.com" title="publications">gmail</a>]

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

Studying
======
  <ul>{% for post in site.teaching %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

Skills
======
<p style="text-align:justify; text-justify:inter-ideograph;">Programming Capability | Innovation and Creativity | Team-Work Ability</p>

Selected Blog Posts
======

<ul>{% for post in site.posts %}
    {% if post.star == "superior" %}
         {% include archive-single.html %}
    {% endif %}
  {% endfor %}</ul>
