{% filter indent(width=4) %}
{% if dims == 2 %}
t = kad_feed({{dims}}, {{batchNum}}, {{channel}});
{% elif dims == 3 %}
t = kad_feed({{dims}}, {{batchNum}}, {{channel}}, {{height}});
{% elif dims == 4 %}
t = kad_feed({{dims}}, {{batchNum}}, {{channel}}, {{height}}, {{width}});
{% endif -%}
t->ext_flag |= KANN_F_IN;
{% endfilter %}