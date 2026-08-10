---
permalink: /
title: "Jianfeng Cai (蔡建峰)"
excerpt: "About me"
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

{% include base_path %}

<section class="home-intro" aria-label="Introduction">
  <p class="home-intro__lead">I am a master's student at the <a href="https://eeis.ustc.edu.cn/main.htm">University of Science and Technology of China</a>, advised by Prof. <a href="http://staff.ustc.edu.cn/~lihq/">Houqiang Li</a> and Prof. <a href="http://staff.ustc.edu.cn/~zhwg/index.html">Wengang Zhou</a> in the Microsoft Key Laboratory of Multimedia Computing and Communication.</p>
  <p class="home-intro__research">My research interests focus on large language models and agents. I am always happy to discuss research questions and potential collaborations.</p>
  <p class="home-intro__seeking">I will graduate in Fall 2027 and am currently seeking job opportunities in LLM post-training and coding agents. If you have a suitable opening, please feel free to reach out.</p>
  <nav class="home-contact" aria-label="Direct contact links">
    <a href="{{ base_path }}/images/wechat.jpg">WeChat</a>
    <a href="{{ base_path }}/images/qq.jpg">QQ</a>
    <a href="mailto:xiaobaicai@mail.ustc.edu.cn">School email</a>
    <a href="mailto:cjf1622613693@gmail.com">Gmail</a>
  </nav>
</section>

<section class="home-section" aria-labelledby="home-publications">
  <h2 class="home-section__title" id="home-publications">Publications</h2>
  {% assign publications = site.publications | sort: "date" | reverse %}
  <ul class="cv-list">
    {% for post in publications limit: 5 %}
      {% include archive-single-cv.html %}
    {% endfor %}
  </ul>
  <p class="home-section__more"><a href="{{ base_path }}/publications/">View all publications <span aria-hidden="true">&rarr;</span></a></p>
</section>

<section class="home-section" aria-labelledby="home-internships">
  <h2 class="home-section__title" id="home-internships">Internships</h2>
  {% assign internships = site.internships | sort: "start_date" | reverse %}
  <div class="internship-list internship-list--compact" role="list" aria-label="Recent internships">
    {% for internship in internships limit: 3 %}
      {% include internship-card.html internship=internship compact=true %}
    {% endfor %}
  </div>
  <p class="home-section__more"><a href="{{ base_path }}/internships/">View all internships <span aria-hidden="true">&rarr;</span></a></p>
</section>

<section class="home-section" aria-labelledby="home-studying">
  <h2 class="home-section__title" id="home-studying">Studying</h2>
  {% assign teaching_items = site.teaching | sort: "date" | reverse %}
  <ul class="cv-list">
    {% for post in teaching_items %}
      {% include archive-single-cv.html %}
    {% endfor %}
  </ul>
</section>

<section class="home-section" aria-labelledby="home-skills">
  <h2 class="home-section__title" id="home-skills">Technical Skills</h2>
  <ul class="skill-list" aria-label="Technical skills">
    <li>Python</li>
    <li>C/C++</li>
    <li>PyTorch</li>
    <li>verl</li>
    <li>OpenRLHF</li>
    <li>DeepSpeed</li>
  </ul>
</section>

<section class="home-section" aria-labelledby="home-blog-posts">
  <h2 class="home-section__title" id="home-blog-posts">Selected Blog Posts</h2>
  {% assign featured_posts = site.posts | where: "star", "superior" %}
  <ul class="cv-list">
    {% for post in featured_posts limit: 5 %}
      {% include archive-single-cv.html %}
    {% endfor %}
  </ul>
  <p class="home-section__more"><a href="{{ base_path }}/year-archive/">View all blog posts <span aria-hidden="true">&rarr;</span></a></p>
</section>
