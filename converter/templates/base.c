#include <stdlib.h>
#include <stdio.h>
#include "ikann.h"
#include "model.h"

kann_t *init_nn() {

	kann_t * nn;
	kad_node_t *t;
	{% for b in codeBlocks %}
	{{b}}
	{%- endfor %}
	t->ext_flag = KANN_F_OUT;
	nn = kann_new(t, 0);
	return nn;
}