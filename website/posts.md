---
layout: default
title: All Posts
permalink: /posts/
---

# Blog Posts

{% for post in site.posts %}
  <div class="post-preview">
    <h2>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    </h2>
    <span class="post-date">{{ post.date | date: "%B %d, %Y" }}</span>
    <p>{{ post.excerpt }}</p>
    <a href="{{ post.url | relative_url }}">Read more</a>
    <hr>
  </div>
{% endfor %} 