---
layout: default
title: Home
---

# Welcome to QHChina Lab

QHChina Lab is dedicated to quantitative research in Chinese humanities, providing computational tools and resources for researchers in the field.

## Latest News and Updates

<div class="posts-list">
  {% for post in site.posts limit:5 %}
    <div class="post-preview">
      <h2>
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </h2>
      <span class="post-date">{{ post.date | date: "%B %d, %Y" }}</span>
      <p>{{ post.excerpt }}</p>
      <a href="{{ post.url | relative_url }}">Read more</a>
    </div>
  {% endfor %}
</div>

{% if site.posts.size > 0 %}
<div class="all-posts">
  <a href="{{ "/posts" | relative_url }}">View all posts</a>
</div>
{% endif %} 