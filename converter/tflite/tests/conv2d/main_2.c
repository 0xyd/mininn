#include <stdlib.h>
#include <stdio.h>
#include "ikann.h"
#include "utils.h"
#include "model.h"

// Data with 1 channels
static float data[] = {

	1., 2., 3., 4., 5.,
	6., 7., 8., 9., 10.,
	11., 12., 13., 14., 15.,
	16., 17., 18., 19., 20.,
	21., 22., 23., 24., 25.,
};

int main(int argc, char *argv[]) {


	FILE *fp;
	kann_t *nn = init_nn();

	int i;
	float *y = kann_apply1(nn, data);

	for (i=0; i<1; i++) printf("%.5f, ", y[i]);
	printf("\n");
	
	fp = fopen("model_output.txt", "w");
	for (int i=0; i<1; i++) fprintf(fp, "%.8f,", y[i]);
	fclose(fp);

	return 0;
}