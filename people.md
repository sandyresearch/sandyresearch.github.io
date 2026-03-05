---
layout: page
title: People
show_title: false
---

<div class="people-section">
  {% if site.data.people.faculty.size > 0 %}
  <h2>Faculty</h2>
  <div class="people-grid">
    {% for person in site.data.people.faculty %}
    <div class="person-card">
      <div class="person-image">
        <img src="{{ person.image | relative_url }}" alt="{{ person.name }}">
      </div>
      <div class="person-name">{{ person.name }}</div>
      <div class="person-links">
        {% if person.links.google_scholar != nil and person.links.google_scholar != "" %}<a href="{{ person.links.google_scholar }}"><i class="svg-icon google-scholar"></i></a>{% endif %}
        {% if person.links.website != nil and person.links.website != "" %}<a href="{{ person.links.website }}"><i class="svg-icon globe"></i></a>{% endif %}
        {% if person.links.github != nil and person.links.github != "" %}<a href="{{ person.links.github }}"><i class="svg-icon github"></i></a>{% endif %}
        {% if person.links.twitter != nil and person.links.twitter != "" %}<a href="{{ person.links.twitter }}"><i class="svg-icon twitter"></i></a>{% endif %}
      </div>
    </div>
    {% endfor %}
  </div>
  {% endif %}

  {% if site.data.people.phd_students.size > 0 %}
  <h2>PhD Students</h2>
  <div class="people-grid">
    {% for person in site.data.people.phd_students %}
    <div class="person-card">
      <div class="person-image">
        <img src="{{ person.image | relative_url }}" alt="{{ person.name }}">
      </div>
      <div class="person-name">{{ person.name }}</div>
      <div class="person-links">
        {% if person.links.google_scholar != nil and person.links.google_scholar != "" %}<a href="{{ person.links.google_scholar }}"><i class="svg-icon google-scholar"></i></a>{% endif %}
        {% if person.links.website != nil and person.links.website != "" %}<a href="{{ person.links.website }}"><i class="svg-icon globe"></i></a>{% endif %}
        {% if person.links.github != nil and person.links.github != "" %}<a href="{{ person.links.github }}"><i class="svg-icon github"></i></a>{% endif %}
        {% if person.links.twitter != nil and person.links.twitter != "" %}<a href="{{ person.links.twitter }}"><i class="svg-icon twitter"></i></a>{% endif %}
      </div>
    </div>
    {% endfor %}
  </div>
  {% endif %}
</div>

