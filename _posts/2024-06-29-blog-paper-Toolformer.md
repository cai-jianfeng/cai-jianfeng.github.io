---
title: 'Toolformer'
date: 24-06-29
update: 24-06-29
permalink: /posts/2024/06/blog-toolformer/
star: superior
tags:
  - 计算机基本知识
---

1. Sampling API Calls: 
   1. Prompt P(x), sample i from 1 ~ n, filter p_i = p_M(<API>|P(x),x_{1:i-1}) > t_s (up to k) -> I.
   2. m API calls, for i in I, using prefix [P(x), x_{1:i-1}, <API>] until generate </API> -> c_i^1 ~ c_i^m
2. Executing API Calls: for each c_i^j -> a single text sequence r_i^j.
3. Filtering API Calls: 
