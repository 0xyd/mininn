#include <stdlib.h>
#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "ikann.h"

#define KAD_FUNC_OP2(fname, op) kad_node_t *fname(kad_node_t *x, kad_node_t *y) { return kad_op2_core((op), x, y); }

KAD_FUNC_OP2(kad_add, 1)
// KAD_FUNC_OP2(kad_sub, 23)
// KAD_FUNC_OP2(kad_mul, 2)
KAD_FUNC_OP2(kad_cmul, 3)
// KAD_FUNC_OP2(kad_matmul, 9)
// KAD_FUNC_OP2(kad_ce_multi, 13)
// KAD_FUNC_OP2(kad_ce_bin, 22)
// KAD_FUNC_OP2(kad_ce_bin_neg, 4)
// KAD_FUNC_OP2(kad_mse, 29)

static inline kad_node_t *kad_finalize_node(kad_node_t *s) /* a helper function */
{
	int i;
	if (kad_op_list[s->op](s, KAD_SYNC_DIM) < 0) { /* check dimension */
		if (s->ptr) free(s->ptr);
		free(s->child); free(s);
		return 0;
	}
	for (i = 0; i < s->n_child; ++i)
		if (kad_is_back(s->child[i]))
			break;
	if (i < s->n_child) s->flag |= KAD_VAR;
	return s;
}

static inline kad_node_t *kad_op2_core(int op, kad_node_t *x, kad_node_t *y)
{
	kad_node_t *s;
	s = kad_new_core(0, op, 2);
	s->child[0] = x, s->child[1] = y;
	return kad_finalize_node(s);
}

static inline kad_node_t *kad_op1_core(int op, kad_node_t *x)
{
	kad_node_t *s;
	s = kad_new_core(0, op, 1);
	s->child[0] = x;
	return kad_finalize_node(s);
}

static inline kad_node_t *kad_vleaf(uint8_t flag, float *x, float *g, int n_d, va_list ap)
{
	int i;
	kad_node_t *p;
	if (n_d > KAD_MAX_DIM) return 0;
	p = (kad_node_t*)calloc(1, sizeof(kad_node_t));
	p->n_d = n_d;

	// Set up the size of the input 
	for (i = 0; i < n_d; ++i)
		p->d[i] = va_arg(ap, int32_t);
	p->x = x, p->g = g, p->flag = flag;
	return p;
}

kad_node_t *kad_feed(int n_d, ...)
{
	printf("in kad_feed \n");
	kad_node_t *p;
	va_list ap;
	va_start(ap, n_d); p = kad_vleaf(0, 0, 0, n_d, ap); va_end(ap);
	return p;
}

kad_node_t *kann_layer_input(int n1)
{
	kad_node_t *t;
	t = kad_feed(2, 1, n1), t->ext_flag |= KANN_F_IN;
	return t;
}

kad_node_t *kann_new_leaf_array(int *offset, kad_node_p *par, uint8_t flag, float x0_01, int n_d, int32_t d[KAD_MAX_DIM])
{
	int i, len, off = offset && par? *offset : -1;
	kad_node_t *p;

	printf("off : %d \n", off);
	printf("offset : %d \n", offset);

	if (off >= 0 && par[off]) return par[(*offset)++];
	p = (kad_node_t*)calloc(1, sizeof(kad_node_t));
	p->n_d = n_d, p->flag = flag;
	memcpy(p->d, d, n_d * sizeof(int32_t));
	len = kad_len(p);
	p->x = (float*)calloc(len, sizeof(float));
	for (i = 0; i < len; ++i)
		p->x[i] = 0.01;
	// if (p->n_d <= 1) {
	// 	for (i = 0; i < len; ++i)
	// 		// p->x[i] = 0.0;
	// 		p->x[i] = 0.01;
	// } else {
	// 	double sdev_inv;
	// 	sdev_inv = 1.0 / sqrt((double)len / p->d[0]);
	// 	for (i = 0; i < len; ++i)
	// 		p->x[i] = (float)(kad_drand_normal(0) * sdev_inv);
	// }
	if (off >= 0) par[off] = p, ++(*offset);
	return p;
}

kad_node_t *kann_new_leaf(uint8_t flag, float x0_01, int n_d, ...)
{
	int32_t i, d[KAD_MAX_DIM];
	va_list ap;
	va_start(ap, n_d); for (i = 0; i < n_d; ++i) d[i] = va_arg(ap, int); va_end(ap);
	return kann_new_leaf_array(0, 0, flag, x0_01, n_d, d);
}


kad_node_t *kann_new_leaf2(int *offset, kad_node_p *par, uint8_t flag, float x0_01, int n_d, ...)
{
	int32_t i, d[KAD_MAX_DIM];
	va_list ap;
	va_start(ap, n_d); for (i = 0; i < n_d; ++i) d[i] = va_arg(ap, int); va_end(ap);
	return kann_new_leaf_array(offset, par, flag, x0_01, n_d, d);
}

kad_node_t *kann_layer_dense(kad_node_t *in, int n1) { return kann_layer_dense2(0, 0, in, n1); }

kad_node_t *kann_layer_dense2(int *offset, kad_node_p *par, kad_node_t *in, int n1)
{
	printf("in kann_layer_dense2;\n");
	int n0;
	kad_node_t *w, *b;
	n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
	// printf("Generating weights for dense \n");
	// printf("n0:%d n1:%d \n", n0, n1);
	w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
	// printf("Generating bias for dense \n");
	b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
	return kad_add(kad_cmul(in, w), b);
}


