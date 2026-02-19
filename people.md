---
layout: page
title: People
---

## PhD Students

<div class="people-grid">
{% for person in site.data.people.phd_students %}
<a href="{{ person.website }}" class="person-card" target="_blank" rel="noopener">
  <img src="{{ person.photo }}" alt="{{ person.name }}">
  <p class="person-name">{{ person.name }}</p>
</a>
{% endfor %}
</div>

## Faculty

<div class="people-grid">
{% for person in site.data.people.faculty %}
<a href="{{ person.website }}" class="person-card" target="_blank" rel="noopener">
  <img src="{{ person.photo }}" alt="{{ person.name }}">
  <p class="person-name">{{ person.name }}</p>
</a>
{% endfor %}
</div>
