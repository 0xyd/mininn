#include <stdlib.h>
#include <stdio.h>
#include "ikann.h"
#include "utils.h"
#include "model.h"


// const static float data[] = {
//     9., 8., 7.,
//     6., 5., 4.,
//     3., 2., 1.,
//     9., 8., 7.,
//     6., 5., 4.,
//     3., 2., 1.,
// };

// Data with two channels
static float data[] = {
    1., 2., 3.,
    4., 5., 6.,
    7., 8., 9.,

    4., 5., 6.,
    1., 2., 3.,
    7., 8., 9.,
};

int main(int argc, char *argv[]) {

	FILE *fp;
	kann_t *nn = init_nn();

	// We have to rotate value for 180 degree since the tensorflow use NHWC but
	// iKann use NCHW
	conv_nhwc_to_ncwh(2, 3, 3, data);	
	for (int i=0; i< 9; i++) {
		printf("%.2f ", data[i]);
		if ((i+1) % 3 == 0 && i>0) printf("\n");
	}
	for (int i=9; i< 18; i++) {
		printf("%.2f ", data[i]);
		if ((i+1) % 3 == 0 && i>0) printf("\n");
	}

	float *y = kann_apply1(nn, data);
	ccccccc
	fp = fopen("model_output.txt", "w");
	for (int i=0; i<10; i++) {
		fprintf(fp, "%.8f,", y[i]);
	}
	fclose(fp);

	return 0;
}