#include <stdlib.h>
#include <stdio.h>
#include "ikann.h"
#include "model.h"

const static float data[] = { 1., 2., 3. };

int main(int argc, char *argv[]) {

	kann_t *nn = init_nn();

	float *y = kann_apply1(nn, data);
	for (int i=0; i<3; i++) printf("y[%d]: %f \n", i, y[i]);

	return 0;
}