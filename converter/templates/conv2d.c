t = kann_layer_conv2d(t, {{ filters }}, {{ kernelH }}, {{ kernelW }}, {{ strideH }}, {{ strideW }}, {{ paddingH }}, {{ paddingW }});
{% filter indent(width=4) %}
{% for w in weights -%}
	t->child[0]->child[1]->x[{{loop.index-1}}] = {{w.item()}};
{% endfor %}
{% for b in bias -%}
	t->child[1]->x[{{loop.index-1}}] = {{b.item()}};
{% endfor %}
{% endfilter %}