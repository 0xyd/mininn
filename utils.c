void conv_nhwc_to_ncwh(int c, int h, int w, float *d)
{
	/*
	c: number of channels

	h: height of data

	w: width of data

	*d: data 
	*/
	int i, j, k, offset=0;

	for (k = 0; k < c; k++) {

		float tmp;
		i = offset; j = offset+h*w-1;
		while(1) {
			if (i>=j) break;
			tmp = d[i]; d[i] = d[j]; d[j] = tmp;
			i++; j--;
			
		}
		offset += h*w;
	}	
}