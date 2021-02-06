#include <stdlib.h>
#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "ikann.h"

static inline void kad_copy_dim1(kad_node_t *dst, const kad_node_t *src) /* set the dimension/shape of dst to src */
{
	dst->n_d = src->n_d;
	if (src->n_d) memcpy(dst->d, src->d, src->n_d * sizeof(int));
}
static inline int kad_len(const kad_node_t *p) /* calculate the size of p->x */
{
	int n = 1, i;
	for (i = 0; i < p->n_d; ++i) n *= p->d[i];
	return n;
}

/********************************
 * Vector and matrix operations *
 ********************************/
static inline float kad_sdot(int n, const float *x, const float *y) /* BLAS sdot */
{
	int i;
	float s = 0.;
	for (i = 0; i < n; ++i) s += x[i] * y[i];
	return s;
}

static inline void kad_saxpy_inlined(int n, float a, const float *x, float *y) // BLAS saxpy
{
	int i;
	for (i = 0; i < n; ++i) y[i] += a * x[i];
}

void kad_vec_mul_sum(int n, float *a, const float *b, const float *c)
{
	int i;
	for (i = 0; i < n; ++i) a[i] += b[i] * c[i];
}

void kad_saxpy(int n, float a, const float *x, float *y) { kad_saxpy_inlined(n, a, x, y); }

void kad_sgemm_simple(int trans_A, int trans_B, int M, int N, int K, const float *A, const float *B, float *C) /* simplified BLAS sgemm */
{
	static const int x = 16;
	int i, j, k;
	if (!trans_A && trans_B) {
		for (i = 0; i < M; i += x) {
			for (j = 0; j < N; j += x) {
				int ii, ie = M < i + x? M : i + x;
				int jj, je = N < j + x? N : j + x;
				for (ii = i; ii < ie; ++ii) { /* loop tiling */
					const float *aii = A + ii * K, *bjj;
					float *cii = C + ii * N;
					for (jj = j, bjj = B + j * K; jj < je; ++jj, bjj += K)
						cii[jj] += kad_sdot(K, aii, bjj);
				}
			}
		}
	} else if (!trans_A && !trans_B) {
		for (i = 0; i < M; ++i)
			for (k = 0; k < K; ++k)
				kad_saxpy_inlined(N, A[i*K+k], &B[k*N], &C[i*N]);
	} else if (trans_A && !trans_B) {
		for (k = 0; k < K; ++k)
			for (i = 0; i < M; ++i)
				kad_saxpy_inlined(N, A[k*M+i], &B[k*N], &C[i*N]);
	} else abort(); /* not implemented for (trans_A && trans_B) */
}

/**********************
 * Graph construction *
 **********************/

static inline kad_node_t *kad_new_core(int n_d, int op, int n_child)
{
	kad_node_t *s;
	if (n_d >= KAD_MAX_DIM) {
		return 0;
	}
	s = (kad_node_t*)calloc(1, sizeof(kad_node_t));
	s->n_d = n_d, s->op = op, s->n_child = n_child;
	if (s->n_child) s->child = (kad_node_t**)calloc(s->n_child, sizeof(kad_node_t*));
	return s;
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
	
	p->x = x, p->flag = flag;
	// gradient is not important in inference
	// p->x = x, p->g = g, p->flag = flag;
	return p;
}

int kad_op_add(kad_node_t *p, int action)
{
	int i, n0, n1;
	kad_node_t *q[2];

	q[0] = p->child[0], n0 = kad_len(q[0]);
	q[1] = p->child[1], n1 = kad_len(q[1]);
	if (action == KAD_SYNC_DIM) {
		if (n0 % n1 != 0) return -1;
		kad_copy_dim1(p, q[0]);
	} else if (action == KAD_FORWARD) {
		assert(n0 >= n1);
		memcpy(p->x, q[0]->x, n0 * sizeof(float));
		for (i = 0; i < n0; i += n1) 
			kad_saxpy(n1, 1.0f, q[1]->x, p->x + i);

	} 
	return 0;
}

/* Operation 2:  element-wise multiplication */

int kad_op_mul(kad_node_t *p, int action)
{
	int i, n0, n1;
	kad_node_t *q[2];

	q[0] = p->child[0], n0 = kad_len(q[0]);
	q[1] = p->child[1], n1 = kad_len(q[1]);
	if (action == KAD_SYNC_DIM) {
		if (n0 % n1 != 0) return -1;
		kad_copy_dim1(p, q[0]);
	} else if (action == KAD_FORWARD) {
		assert(n0 >= n1);
		memset(p->x, 0, n0 * sizeof(float));
		if (q[0]->x != 0 && q[1]->x != 0)
			for (i = 0; i < n0; i += n1) /* TODO: optimize when n1==1 */
				kad_vec_mul_sum(n1, p->x + i, q[0]->x + i, q[1]->x);
	}
	return 0;
}

int kad_op_cmul(kad_node_t *p, int action)
{
	int i, n_a_row, n_b_row, n_col, n_a_col = 1, n_b_col = 1;
	kad_node_t *q[2];
	q[0] = p->child[0], q[1] = p->child[1];
	n_col = q[0]->d[q[0]->n_d - 1] > q[1]->d[q[1]->n_d - 1]? q[0]->d[q[0]->n_d - 1] : q[1]->d[q[1]->n_d - 1];
	for (i = q[0]->n_d - 1; i >= 0; --i) if (n_a_col < n_col) n_a_col *= q[0]->d[i];
	for (i = q[1]->n_d - 1; i >= 0; --i) if (n_b_col < n_col) n_b_col *= q[1]->d[i];
	n_a_row = kad_len(q[0]) / n_a_col, n_b_row = kad_len(q[1]) / n_b_col;
	if (action == KAD_SYNC_DIM) {
		if (n_a_col != n_b_col) return -1;
		p->n_d = 2, p->d[0] = n_a_row, p->d[1] = n_b_row;
	} else if (action == KAD_FORWARD) {
		memset(p->x, 0, n_a_row * n_b_row * sizeof(float));
		if (q[0]->x && q[1]->x)
			kad_sgemm_simple(0, 1, n_a_row, n_b_row, n_col, q[0]->x, q[1]->x, p->x); /* Y = X * trans(W) */
	} 
	return 0;
}

/********** Activation functions **********/

int kad_op_sigm(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i)
			p->x[i] = 1.0f / (1.0f + expf(-q->x[i]));
	} 
	return 0;
}

int kad_op_tanh(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i) {
			if (q->x[i] < -20.0f) p->x[i] = -1.0f;
			else {
				float y;
				y = expf(-2.0f * q->x[i]);
				p->x[i] = (1.0f - y) / (1.0f + y);
			}
		}
	} 
	return 0;
}

int kad_op_relu(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i) 
			p->x[i] = q->x[i] > 0.0f? q->x[i] : 0.0f;
	} 
	return 0;
}

int kad_op_sin(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0];
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i) p->x[i] = sinf(q->x[i]);
	} 
	return 0;
}

int kad_op_softmax(kad_node_t *p, int action)
{
	int i, j, n1, d0;
	kad_node_t *q = p->child[0];

	n1 = q->d[q->n_d - 1];
	d0 = kad_len(q) / n1;
	if (action == KAD_SYNC_DIM) {
		kad_copy_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (j = 0; j < d0; ++j) {
			float s, max, *x = &q->x[j * n1], *y = &p->x[j * n1];
			for (i = 0, max = -FLT_MAX; i < n1; ++i)
				max = max > x[i]? max : x[i];
			for (i = 0, s = 0.0f; i < n1; ++i) {
				y[i] = expf(x[i] - max);
				s += y[i];
			}
			for (i = 0, s = 1.0f / s; i < n1; ++i) y[i] *= s;
		}
	} 
	return 0;
}

// Convolutional operation
/********** 2D convolution **********/
typedef struct {
	int kernel_size, stride, pad[2];
} conv_conf_t;

static void conv_rot180(int d0, int d1, float *x) /* rotate/reverse a weight martix */
{
	int i, j;
	for (i = 0; i < d0; ++i) {
		float tmp, *xi = &x[i * d1];
		for (j = 0; j < d1>>1; ++j)
			tmp = xi[j], xi[j] = xi[d1-1-j], xi[d1-1-j] = tmp; 
	}
}

static void conv2d_move_1to3(int d[4], const float *x, float *y) /* convert the NCHW shape to the NHWC shape */
{
	int i, j, k, l;
	for (i = 0; i < d[0]; ++i)
		for (j = 0; j < d[1]; ++j)
			for (k = 0; k < d[2]; ++k) {
				int ik = (i * d[2] + k) * d[3], ijk = ((i * d[1] + j) * d[2] + k) * d[3];
				for (l = 0; l < d[3]; ++l)
					y[(ik + l) * d[1] + j] = x[ijk + l];
			}
}

static void conv2d_add_3to1(int d[4], const float *y, float *x) /* convert the NHWC shape back to NCHW and add to another NCHW-shaped array */
{
	int i, j, k, l;
	for (i = 0; i < d[0]; ++i)
		for (j = 0; j < d[1]; ++j)
			for (k = 0; k < d[2]; ++k) {
				int ik = (i * d[2] + k) * d[3], ijk = ((i * d[1] + j) * d[2] + k) * d[3];
				for (l = 0; l < d[3]; ++l)
					x[ijk + l] += y[(ik + l) * d[1] + j];
			}
}

#define conv_out_size(in_size, aux) (((in_size) - (aux)->kernel_size + (aux)->pad[0] + (aux)->pad[1]) / (aux)->stride + 1)

#define process_row_for(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
	int j, l; \
	if (_stride > 1) { \
		for (l = 0; l < _wn; ++l) { \
			const float *xl = &_xx[l - _pad]; \
			for (j = 0; j < _pn; ++j, xl += _stride) _t[j] = *xl; \
			kad_saxpy(_pn, _ww[l], _t, _yy); \
		} \
	} else for (l = 0; l < _wn; ++l) kad_saxpy(_pn, _ww[l], &_xx[l - _pad], _yy); \
} while (0)

#define process_row_back_x(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
	int j, l; \
	if (_stride > 1) { \
		for (l = 0; l < _wn; ++l) { \
			float *xl = &_xx[l - _pad]; \
			memset(_t, 0, _pn * sizeof(float)); \
			kad_saxpy(_pn, _ww[l], _yy, _t); \
			for (j = 0; j < _pn; ++j, xl += _stride) *xl += _t[j]; \
		} \
	} else for (l = 0; l < _wn; ++l) kad_saxpy(_pn, _ww[l], _yy, &_xx[l - _pad]); \
} while (0)

#define process_row_back_w(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
	int j, l; \
	if (_stride > 1) { \
		for (l = 0; l < _wn; ++l) { \
			const float *xl = &_xx[l - _pad]; \
			for (j = 0; j < _pn; ++j, xl += _stride) _t[j] = *xl; \
			_ww[l] += kad_sdot(_pn, _yy, _t); \
		} \
	} else for (l = 0; l < _wn; ++l) _ww[l] += kad_sdot(_pn, _yy, &_xx[l - _pad]); \
} while (0)
int kad_op_conv2d(kad_node_t *p, int action) /* in the number-channel-height-width (NCHW) shape */
{
#define conv2d_loop1(_x, _w, _y, _tmp, _row_func) do { /* for the NCHW shape */ \
		int n, c1, c0, i, k, ii; \
		for (n = 0; n < q->d[0]; ++n) /* mini-batch */ \
			for (c1 = 0; c1 < w->d[0]; ++c1) /* output channel */ \
				for (c0 = 0; c0 < w->d[1]; ++c0) /* input channel */ \
					for (k = 0; k < w->d[2]; ++k) { /* kernel row */ \
						float *_ww = &(_w)[((c1 * w->d[1] + c0) * w->d[2] + k) * w->d[3]]; \
						for (i = 0, ii = k - aux[0].pad[0]; i < p->d[2] && ii >= 0 && ii < q->d[2]; ++i, ii += aux[0].stride) { /* output row */ \
							float *_xx = &(_x)[((n * q->d[1] + c0) * q->d[2] + ii) * q->d[3]]; \
							float *_yy = &(_y)[((n * p->d[1] + c1) * p->d[2] + i)  * p->d[3]]; \
							if (x_padded) { \
								memcpy(x_padded + aux[1].pad[0], _xx, q->d[3] * sizeof(float)); \
								_xx = x_padded + aux[1].pad[0]; \
							} \
							_row_func(_xx, _ww, _yy, w->d[3], p->d[3], aux[1].stride, aux[1].pad[0], (_tmp)); \
						} /* ~i */ \
					} /* ~k, c0, c1, n */ \
	} while (0)

#define conv2d_loop2(_x, _w, _y, _code) do { /* for the NHWC shape */ \
		int n, c1, i, j, k, ii, j_skip = aux[1].stride * q->d[1], m = w->d[3] * w->d[1]; \
		for (n = 0; n < q->d[0]; ++n) /* mini-batch */ \
			for (c1 = 0; c1 < w->d[0]; ++c1) /* output channel */ \
				for (k = 0; k < w->d[2]; ++k) { /* kernel row */ \
					float *_ww = &(_w)[(c1 * w->d[2] + k) * m]; \
					for (i = 0, ii = k - aux[0].pad[0]; i < p->d[2] && ii >= 0 && ii < q->d[2]; ++i, ii += aux[0].stride) { /* output and input row */ \
						float *_xx = &(_x)[(n * q->d[2] + ii) * q->d[3] * q->d[1]]; \
						float *_yy = &(_y)[((n * p->d[1] + c1) * p->d[2] + i) * p->d[3]]; \
						if (x_padded) { \
							memcpy(x_padded + aux[1].pad[0] * q->d[1], _xx, q->d[3] * q->d[1] * sizeof(float)); \
							_xx = x_padded; \
						} \
						for (j = 0; j < p->d[3]; ++j, _xx += j_skip, ++_yy) _code; /* output and input column */ \
					} /* ~i */ \
				} /* ~k, c1, n */ \
	} while (0)

	conv_conf_t *aux = (conv_conf_t*)p->ptr;
	kad_node_t *q = p->child[0], *w = p->child[1];
	float *t = 0, *q1 = 0, *w1 = 0, *x_padded = 0;
	int algo_switch = 0;

	if (action == KAD_FORWARD) { /* allocate working space */
		if (w->d[3] * w->d[1] < 16) {
			t = (float*)malloc(p->d[3] * sizeof(float));
			x_padded = aux[1].pad[0] + aux[1].pad[1] > 0? (float*)calloc(q->d[3] + aux[1].pad[0] + aux[1].pad[1], sizeof(float)) : 0;
		} else {
			q1 = (float*)malloc(kad_len(q) * sizeof(float));
			w1 = (float*)malloc(kad_len(w) * sizeof(float));
			x_padded = aux[1].pad[0] + aux[1].pad[1] > 0? (float*)calloc((q->d[3] + aux[1].pad[0] + aux[1].pad[1]) * q->d[1], sizeof(float)) : 0;
			algo_switch = 1;
		}
	}
	if (action == KAD_SYNC_DIM) {
		if (q->n_d != 4 || w->n_d != 4) return -1;
		if (q->d[1] != w->d[1]) return -1; /* unmatched input channels */
		p->n_d = 4;
		p->d[0] = q->d[0], p->d[1] = w->d[0], p->d[2] = conv_out_size(q->d[2], &aux[0]), p->d[3] = conv_out_size(q->d[3], &aux[1]);
	} else if (action == KAD_FORWARD) {
		conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
		memset(p->x, 0, kad_len(p) * sizeof(float));
		if (!algo_switch) { /* this is the first algorithm */
			conv2d_loop1(q->x, w->x, p->x, t, process_row_for);
		} else { /* this is the second algorithm */
			conv2d_move_1to3(q->d, q->x, q1);
			conv2d_move_1to3(w->d, w->x, w1);
			conv2d_loop2(q1, w1, p->x, (*_yy += kad_sdot(m, _ww, _xx)));
		}
		conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
	}
	free(t); free(q1); free(w1); free(x_padded);
	return 0;
}

int kad_op_max2d(kad_node_t *p, int action)
{
	conv_conf_t *aux = (conv_conf_t*)p->ptr;
	kad_node_t *q = p->child[0];
	if (action == KAD_SYNC_DIM) {
		if (q->n_d != 4) return -1;
		p->n_d = 4;
		p->d[0] = q->d[0], p->d[1] = q->d[1], p->d[2] = conv_out_size(q->d[2], &aux[0]), p->d[3] = conv_out_size(q->d[3], &aux[1]);
	} else if (action == KAD_ALLOC) {
		// p->gtmp = realloc(p->gtmp, kad_len(p) * sizeof(int));
	} else if (action == KAD_FORWARD) {
		int rest = 1, len, t, i;
		// int *f = (int*)p->gtmp;
		len = kad_len(p);
		for (i = 0; i < len; ++i) p->x[i] = -FLT_MAX;
		for (i = 0; i < p->n_d - 2; ++i) rest *= p->d[i];
		for (t = 0; t < rest; ++t) {
			int i, j, k, l, p_row = p->d[p->n_d - 2], p_col = p->d[p->n_d - 1];
			for (i = 0; i < p_row; ++i) {
				int u = (t * p_row + i) * p_col;
				for (k = 0; k < aux[0].kernel_size; ++k) {
					int v, v0, v_end, ii = i * aux[0].stride + k - aux[0].pad[0];
					if (ii < 0 || ii >= q->d[p->n_d - 2]) continue;
					v0 = (t * q->d[p->n_d - 2] + ii) * q->d[p->n_d - 1];
					v_end = v0 + q->d[p->n_d - 1];
					for (l = 0; l < aux[1].kernel_size; ++l)
						for (j = 0, v = v0 + (l > aux[1].pad[0]? l - aux[1].pad[0] : 0); j < p_col && v < v_end; ++j, v += aux[1].stride)
							if (p->x[u + j] < q->x[v])
								p->x[u + j] = q->x[v];
								// p->x[u + j] = q->x[v], f[u + j] = v;
				} /* ~k */
			} /* ~i */
		}
	} 
	return 0;
}

/*******************************/

kad_op_f kad_op_list[KAD_MAX_OP] = {
	0,
	kad_op_add,        /* 1:  element-wise addition */
	kad_op_mul,        /* 2:  element-wise multiplication */
	kad_op_cmul,       /* 3:  column multiplication */
	kad_op_sigm,	   /* 4: */
	kad_op_tanh,	   /* 5: */
	kad_op_relu,	   /* 6: */
	kad_op_softmax,    /* 7: */
	kad_op_conv2d, 	   /* 8: 2D Concoluional */
	kad_op_max2d,      /* 9: 2D Maxpooling */
};

static inline kad_node_t *kad_finalize_node(kad_node_t *s)  // a helper function 
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

#define KAD_FUNC_OP1(fname, op) kad_node_t *fname(kad_node_t *x) { return kad_op1_core((op), x); }
KAD_FUNC_OP1(kad_sigm, 4)
KAD_FUNC_OP1(kad_tanh, 5)
KAD_FUNC_OP1(kad_relu, 6)
KAD_FUNC_OP1(kad_softmax, 7)


// This mechanism is not suitable for an inference library
#define KAD_FUNC_OP2(fname, op) kad_node_t *fname(kad_node_t *x, kad_node_t *y) { return kad_op2_core((op), x, y); }

KAD_FUNC_OP2(kad_add, 1)
KAD_FUNC_OP2(kad_mul, 2)
KAD_FUNC_OP2(kad_cmul, 3)

/********** Multi-node pooling **********/

static kad_node_t *kad_pooling_general(int op, int n, kad_node_t **x)
{
	int i;
	kad_node_t *s;
	s = kad_new_core(0, op, n);
	s->flag |= KAD_POOL;
	for (i = 0; i < n; ++i)
		s->child[i] = x[i];
	return kad_finalize_node(s);
}

kad_node_t *kad_avg(int n, kad_node_t **x)   { return kad_pooling_general(10, n, x); }

/********************/

kad_node_t *kad_feed(int n_d, ...)
{
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

	if (off >= 0 && par[off]) return par[(*offset)++];
	p = (kad_node_t*)calloc(1, sizeof(kad_node_t));
	p->n_d = n_d, p->flag = flag;
	memcpy(p->d, d, n_d * sizeof(int32_t));
	len = kad_len(p);
	p->x = (float*)calloc(len, sizeof(float));
	if (p->n_d <= 1) {
		for (i = 0; i < len; ++i)
			p->x[i] = x0_01;
	} else {
		float sdev_inv;
		sdev_inv = 1.0 / sqrt((float)len / p->d[0]);
		for (i = 0; i < len; ++i) 
			p->x[i] = 0.01;
	}
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
	int n0;
	kad_node_t *w, *b;
	n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
	w = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 2, n1, n0);
	b = kann_new_leaf2(offset, par, KAD_VAR, 0.0f, 1, n1);
	return kad_add(kad_cmul(in, w), b);
}

/********** Convolution **********/

/* compute output dimension and padding sizes on both sides */
static inline int conv_find_par(int in_size, int kernel_size, int stride, int pad0, int *new_pad0, int *new_pad1)
{
	int out_size, pad_both;
	/* key equation: out_size = (in_size - kernel_size + pad_both) / stride + 1 */
	if (pad0 == KAD_PAD_SAME && stride == 1) out_size = in_size;
	else out_size = (in_size - kernel_size + (pad0 > 0? pad0 : 0) + stride - 1) / stride + 1;
	pad_both = (out_size - 1) * stride + kernel_size - in_size;
	*new_pad0 = pad_both / 2;
	*new_pad1 = pad_both - *new_pad0;
	return out_size;
}



static inline conv_conf_t *conv2d_gen_aux(int in_row, int in_col, int kernel_r, int kernel_c, int stride_r, int stride_c, int top_pad, int left_pad)
{
	conv_conf_t *cnn;
	cnn = (conv_conf_t*)calloc(2, sizeof(conv_conf_t));
	cnn[0].kernel_size = kernel_r, cnn[0].stride = stride_r;
	cnn[1].kernel_size = kernel_c, cnn[1].stride = stride_c;
	conv_find_par(in_row, kernel_r, stride_r, top_pad,  &cnn[0].pad[0], &cnn[0].pad[1]);
	conv_find_par(in_col, kernel_c, stride_c, left_pad, &cnn[1].pad[0], &cnn[1].pad[1]);
	return cnn;
}

kad_node_t *kad_conv2d(kad_node_t *x, kad_node_t *w, int stride_r, int stride_c, int top_pad, int left_pad)
{
	kad_node_t *s;
	if (x->n_d != 4 || w->n_d != 4) return 0;
	s = kad_new_core(0, 8, 2);
	s->child[0] = x, s->child[1] = w;
	s->ptr = conv2d_gen_aux(x->d[2], x->d[3], w->d[2], w->d[3], stride_r, stride_c, top_pad, left_pad);
	s->ptr_size = sizeof(conv_conf_t) * 2;
	return kad_finalize_node(s);
}

kad_node_t *kad_max2d(kad_node_t *x, int kernel_r, int kernel_c, int stride_r, int stride_c, int top_pad, int left_pad)
{
	kad_node_t *s;
	if (x->n_d != 4) return 0;
	s = kad_new_core(0, 9, 1);
	s->child[0] = x;
	s->ptr = conv2d_gen_aux(x->d[2], x->d[3], kernel_r, kernel_c, stride_r, stride_c, top_pad, left_pad);
	s->ptr_size = sizeof(conv_conf_t) * 2;
	return kad_finalize_node(s);
}

kad_node_t *kann_new_weight_conv2d(int n_out, int n_in, int k_row, int k_col) { return kann_new_leaf(KAD_VAR, 0.0f, 4, n_out, n_in, k_row, k_col); }

kad_node_t *kann_layer_conv2d(kad_node_t *in, int n_flt, int k_rows, int k_cols, int stride_r, int stride_c, int pad_r, int pad_c)
{
	kad_node_t *w;
	w = kann_new_weight_conv2d(n_flt, in->d[1], k_rows, k_cols);
	return kad_conv2d(in, w, stride_r, stride_c, pad_r, pad_c);
}

/***********************
 * Graph linearization *
 ***********************/

static void kad_mark_back(int n, kad_node_t **v)
{
	int i, j;
	for (i = 0; i < n; ++i) {
		if (v[i]->n_child == 0) continue;
		for (j = 0; j < v[i]->n_child; ++j)
			if (kad_is_back(v[i]->child[j]))
				break;
		if (j < v[i]->n_child) v[i]->flag |= KAD_VAR;
		else v[i]->flag &= ~KAD_VAR;
	}
}

static void kad_allocate_internal(int n, kad_node_t **v)
{
	int i;
	kad_mark_back(n, v);
	for (i = 0; i < n; ++i) {
		kad_node_t *p = v[i];
		if (p->n_child == 0) continue;
		p->x = (float*)realloc(p->x, kad_len(p) * sizeof(float));
		// gradient is no need for inference
		// if (kad_is_back(p)) {
		// 	p->g = (float*)realloc(p->g, kad_len(p) * sizeof(float));
		// 	kad_op_list[p->op](p, KAD_ALLOC);
		// }
	}
}

int kad_sync_dim(int n, kad_node_t **v, int batch_size)
{
	int i, req_alloc = 0, req_sync = 0, old_size = 0;
	for (i = 0; i < n; ++i) {
		if (kad_is_feed(v[i])) {
			old_size = v[i]->d[0]; /* TODO: check if all feeds have the same batch size */
			if (batch_size > 0 && v[i]->d[0] != batch_size)
				v[i]->d[0] = batch_size, req_sync = 1;
		} else if (v[i]->n_child > 0 && req_sync)
			kad_op_list[v[i]->op](v[i], KAD_SYNC_DIM);
	}
	if (old_size < batch_size) req_alloc = 1;
	for (i = 0; i < n; ++i)
		if (v[i]->n_child > 0 && v[i]->x == 0) req_alloc = 1;
	if (req_alloc) kad_allocate_internal(n, v);
	return batch_size > 0? batch_size : old_size;
}

#define kvec_t(type) struct { size_t n, m; type *a; }

#define kv_pop(v) ((v).a[--(v).n])

#define kv_push(type, v, x) do { \
		if ((v).n == (v).m) { \
			(v).m = (v).m? (v).m<<1 : 2; \
			(v).a = (type*)realloc((v).a, sizeof(type) * (v).m); \
		} \
		(v).a[(v).n++] = (x); \
	} while (0)

/* IMPORTANT: kad_node_t::tmp MUST BE set to zero before calling this function */
kad_node_t **kad_compile_array(int *n_node, int n_roots, kad_node_t **roots)
{
	int i;
	kvec_t(kad_node_p) stack = {0,0,0}, a = {0,0,0};

	/* generate kad_node_t::tmp, the count of the parent nodes; shifted by 1; lowest bit to detect fake roots */
	for (i = 0; i < n_roots; ++i) {
		roots[i]->tmp = 1; /* mark the root */
		kv_push(kad_node_p, stack, roots[i]);
	}
	while (stack.n) {
		kad_node_t *p = kv_pop(stack);
		for (i = 0; i < p->n_child; ++i) {
			kad_node_t *q = p->child[i];
			if (q->tmp == 0) kv_push(kad_node_p, stack, q);
			q->tmp += 1<<1;
		}
	}

	/* topological sorting (Kahn's algorithm) */
	for (i = 0; i < n_roots; ++i)
		if (roots[i]->tmp>>1 == 0) /* if roots[i]->tmp>>1 != 0, it is not a real root */
			kv_push(kad_node_p, stack, roots[i]);
	while (stack.n) {
		kad_node_t *p = kv_pop(stack);
		kv_push(kad_node_p, a, p);
		for (i = 0; i < p->n_child; ++i) {
			p->child[i]->tmp -= 1<<1;
			if (p->child[i]->tmp>>1 == 0)
				kv_push(kad_node_p, stack, p->child[i]);
		}
	}
	free(stack.a);
	for (i = 0; i < (int)a.n; ++i) { /* check cycles; no cycles if constructed with kad_add() etc */
		assert(a.a[i]->tmp>>1 == 0);
		a.a[i]->tmp = 0;
	}

	/* reverse */
	for (i = 0; i < (int)a.n>>1; ++i) { /* reverse a.a[] */
		kad_node_p t;
		t = a.a[i], a.a[i] = a.a[a.n-1-i], a.a[a.n-1-i] = t;
	}
	kad_allocate_internal(a.n, a.a);

	*n_node = a.n;
	return a.a;
}

/************************************
 * Miscellaneous on compiled graphs *
 ************************************/

int kad_size_var(int n, kad_node_t *const* v)
{
	int c, i;
	for (i = c = 0; i < n; ++i)
		if (kad_is_var(v[i]))
			c += kad_len(v[i]);
	return c;
}

int kad_size_const(int n, kad_node_t *const* v)
{
	int c, i;
	for (i = c = 0; i < n; ++i)
		if (kad_is_const(v[i]))
			c += kad_len(v[i]);
	return c;
}

/**********************************
 * Computate values and gradients *
 **********************************/

static void kad_propagate_marks(int n, kad_node_t **a)
{
	int i, j;
	for (i = n - 1; i >= 0; --i) {
		kad_node_t *p = a[i];
		if (p->tmp > 0) {
			if (kad_is_switch(p)) {
				int32_t *aux = (int32_t*)p->ptr;
				if (p->child[*aux]->tmp == 0)
					p->child[*aux]->tmp = 1;
			} else {
				for (j = 0; j < p->n_child; ++j)
					if (p->child[j]->tmp == 0)
						p->child[j]->tmp = 1;
			}
		}
	}
}

void kad_eval_marked(int n, kad_node_t **a)
{
	int i;
	kad_propagate_marks(n, a);
	for (i = 0; i < n; ++i) {
		if (a[i]->n_child && a[i]->tmp > 0) {
			kad_op_list[a[i]->op](a[i], KAD_FORWARD);
		}
	}
	for (i = 0; i < n; ++i) a[i]->tmp = 0;
}

const float *kad_eval_at(int n, kad_node_t **a, int from)
{
	int i;
	if (from < 0 || from >= n) from = n - 1;
	for (i = 0; i < n; ++i) a[i]->tmp = (i == from);
	kad_eval_marked(n, a);
	return a[from]->x;
}

/**************************************/

#define chk_flg(flag, mask) ((mask) == 0 || ((flag) & (mask)))
#define chk_lbl(label, query) ((query) == 0 || (label) == (query))

int kann_find(const kann_t *a, uint32_t ext_flag, int32_t ext_label)
{
	int i, k, r = -1;
	for (i = k = 0; i < a->n; ++i)
		if (chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
			++k, r = i;
	return k == 1? r : k == 0? -1 : -2;
}

int kann_feed_bind(kann_t *a, uint32_t ext_flag, int32_t ext_label, float **x)
{
	int i, k;
	if (x == 0) return 0;
	for (i = k = 0; i < a->n; ++i)
		if (kad_is_feed(a->v[i]) && chk_flg(a->v[i]->ext_flag, ext_flag) && chk_lbl(a->v[i]->ext_label, ext_label))
			a->v[i]->x = x[k++];
	return k;
}

/******************************************
 *** @@BASIC: fundamental KANN routines ***
 ******************************************/

// Gradient is no need for inference
// static void kad_ext_collate(int n, kad_node_t **a, float **_x, float **_g, float **_c)
static void kad_ext_collate(int n, kad_node_t **a, float **_x, float **_c)
{
	int i, j, k, l, n_var;
	float *x, *g, *c;
	n_var = kad_size_var(n, a);
	x = *_x = (float*)realloc(*_x, n_var * sizeof(float));
	// g = *_g = (float*)realloc(*_g, n_var * sizeof(float));
	c = *_c = (float*)realloc(*_c, kad_size_const(n, a) * sizeof(float));
	// memset(g, 0, n_var * sizeof(float));
	for (i = j = k = 0; i < n; ++i) {
		kad_node_t *v = a[i];
		if (kad_is_var(v)) {
			l = kad_len(v);
			memcpy(&x[j], v->x, l * sizeof(float));
			free(v->x);
			v->x = &x[j];
			// v->g = &g[j];
			j += l;
		} else if (kad_is_const(v)) {
			l = kad_len(v);
			memcpy(&c[k], v->x, l * sizeof(float));
			free(v->x);
			v->x = &c[k];
			k += l;
		}
	}
}

static void kad_ext_sync(int n, kad_node_t **a, float *x, float *g, float *c)
{
	int i, j, k;
	for (i = j = k = 0; i < n; ++i) {
		kad_node_t *v = a[i];
		if (kad_is_var(v)) {
			v->x = &x[j];
			// v->g = &g[j];
			j += kad_len(v);
		} else if (kad_is_const(v)) {
			v->x = &c[k];
			k += kad_len(v);
		}
	}
}

kann_t *kann_new(kad_node_t *cost, int n_rest, ...)
{
	kann_t *a;
	int i, n_roots = 1 + n_rest, has_pivot = 0, has_recur = 0;
	kad_node_t **roots;
	va_list ap;

	// Because we don't need cost function, 
	// instead, we need activation function so that we can output
	// the value.
	// if (cost->n_d != 0) return 0;

	va_start(ap, n_rest);
	roots = (kad_node_t**)malloc((n_roots + 1) * sizeof(kad_node_t*));
	for (i = 0; i < n_rest; ++i)
		roots[i] = va_arg(ap, kad_node_t*);
	roots[i++] = cost;
	va_end(ap);

	cost->ext_flag |= KANN_F_COST;
	a = (kann_t*)calloc(1, sizeof(kann_t));
	a->v = kad_compile_array(&a->n, n_roots, roots);

	for (i = 0; i < a->n; ++i) {
		if (a->v[i]->pre) has_recur = 1;
		if (kad_is_pivot(a->v[i])) has_pivot = 1;
	}
	if (has_recur && !has_pivot) { /* an RNN that doesn't have a pivot; then add a pivot on top of cost and recompile */
		cost->ext_flag &= ~KANN_F_COST;
		roots[n_roots-1] = cost = kad_avg(1, &cost), cost->ext_flag |= KANN_F_COST;
		free(a->v);
		a->v = kad_compile_array(&a->n, n_roots, roots);
	}

	// Remove gradient pointer because it is not necessary
	kad_ext_collate(a->n, a->v, &a->x, &a->c);
	// kad_ext_collate(a->n, a->v, &a->x, &a->g, &a->c);
	free(roots);
	return a;
}

const float *kann_apply1(kann_t *a, float *x)
{
	int i_out;
	i_out = kann_find(a, KANN_F_OUT, 0);
	if (i_out < 0) return 0;
	kann_set_batch_size(a, 1);
	kann_feed_bind(a, KANN_F_IN, 0, &x);
	kad_eval_at(a->n, a->v, i_out);
	return a->v[i_out]->x;
}

