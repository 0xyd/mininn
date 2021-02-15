#include <stdlib.h>
#include <stdio.h>
#include "ikann.h"
#include "utils.h"
#include "model.h"

// Data with 3 channels
static float data[] = {
    1., 2., 3., 4., 5.,
    6., 7., 8., 9., 10.,
    11., 12., 13., 14., 15.,
    16., 17., 18., 19., 20.,
    21., 22., 23., 24., 25.,

    6., 7., 8., 9., 10.,
    1., 2., 3., 4., 5.,
    11., 12., 13., 14., 15.,
    21., 22., 23., 24., 25.,
    16., 17., 18., 19., 20.,

    11., 12., 13., 14., 15.,
    6., 7., 8., 9., 10.,
    1., 2., 3., 4., 5.,
    16., 17., 18., 19., 20.,
    21., 22., 23., 24., 25.,
};

int main(int argc, char *argv[]) {

	FILE *fp;
	kann_t *nn = init_nn();

	float *y = kann_apply1(nn, data);

	for (int i=0; i<27; i++) {
		printf("%.2f,", y[i]);	
		if ((i+1) % 3 == 0 && i>0) printf("\n");
	}
	
	fp = fopen("model_output.txt", "w");
	for (int i=0; i<27; i++) {
		fprintf(fp, "%.2f,", y[i]);
	}
	fclose(fp);

	return 0;
}