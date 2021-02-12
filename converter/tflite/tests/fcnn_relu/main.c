#include <stdlib.h>
#include <stdio.h>
#include "ikann.h"
#include "model.h"

const static float data[] = { 1., 2., 3., 4., 5. };

int main(int argc, char *argv[]) {

	FILE *fp;
	kann_t *nn = init_nn();

	float *y = kann_apply1(nn, data);
	fp = fopen("model_output.txt", "w");
	for (int i=0; i<5; i++) {
		fprintf(fp, "%.8f,", y[i]);
	}
	fclose(fp);

	return 0;
}