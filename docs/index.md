---
layout: default
title: Home
---

<style>
.welcome-section {
  margin: 3rem 0;
  text-align: center;
}

.welcome-section h1 {
  margin-bottom: 1rem;
}

.lead {
  font-size: 1.25rem;
  font-weight: 300;
  margin-bottom: 1rem;
}

.status {
  color: #666;
  font-style: italic;
}
</style>

<div class="welcome-section">
  <h1>Welcome to qh · china</h1>
  <p class="lead">A place where quantitative conceptuality meets Chinese humanities.</p>
  <p class="status">[under construction]</p>
</div>

## What is qh?

"Quantitative Humanities" is a set of scholarly disciplines which can be characterized by the following keywords:

- **Experimentation**: probing literary and visual texts with digital tools and statistical methods
- **Text-centrism**: considering literary artifacts as more than mere anecdotes reflecting extra-textual forces and ideologies
- **Quantification**: reinterpreting narrative phenomena as vectors, frequencies, distances, and networks
- **Big Data & Small Data**: reading large corpora, single novels, and tiny paragraphs
- **Multidisciplinarity**: combining natural language processing with cognitive neuroscience and traditional humanistic methodologies: close reading, genealogy, structuralism, aesthetic theory, etc.
- **Limits of computation**: identifying what stories can tell us about the world and ourselves that matrix multiplications cannot

## Latest Updates

<div class="posts-list">
  {% for post in site.posts limit:5 %}
    <div class="post-preview">
      <h2>
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </h2>
      <span class="post-date">{{ post.date | date: "%B %d, %Y" }}</span>
      <p>{{ post.excerpt }}</p>
      <a href="{{ post.url | relative_url }}" class="read-more">Read more</a>
    </div>
  {% endfor %}
</div>

{% if site.posts.size > 0 %}
<div class="all-posts">
  <a href="{{ "/posts" | relative_url }}">View all posts</a>
</div>
{% endif %} 

## Random Quote

{% include random_quote.html %}