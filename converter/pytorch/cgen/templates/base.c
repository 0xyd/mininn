#include <stdlib.h>
#include <stdio.h>
#include "ikann.h"

kann_t *init_nn() {

	kann_t * nn;
	kad_node_t *t;
	{% for b in codeBlocks %}
	{{b}}
	{%- endfor %}
	return nn;
}