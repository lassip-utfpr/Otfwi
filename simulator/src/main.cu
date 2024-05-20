#include "regression.h"

int main()
{
    int X, Z, T, en_out, n_source, n_sensor;
    int *pos_source_x, *pos_source_z, *pos_sensor_x, *pos_sensor_z;
    float *cquad, *initial, *source, *recording;

    Z = X = 256;
    T = 1200;
    en_out = 0;
    n_source = 1;
    n_sensor = 256;

    pos_source_x = (int*)malloc(n_source*sizeof(int));
    pos_source_z = (int*)malloc(n_source*sizeof(int));
    pos_sensor_x = (int*)malloc(n_sensor*sizeof(int));
    pos_sensor_z = (int*)malloc(n_sensor*sizeof(int));

    cquad = (float*)malloc(X*Z*sizeof(float));
    initial = (float*)malloc(2*X*Z*sizeof(float));
    source = (float*)malloc(n_source*T*sizeof(float));
    recording = (float*)malloc(n_sensor*T*sizeof(float));

    /////////////////////////////////////////////

    pos_source_x[0] = 128;
    pos_source_z[0] = 32;

    for(int i=0;i<n_sensor/2;i++)
    {
	pos_sensor_z[i] = 32;
	pos_sensor_x[i] = X/4 + i;
	pos_sensor_z[i+n_sensor/2] = Z-32;
	pos_sensor_x[i+n_sensor/2] = X/4 + i;
    }

    for(int i=0;i<X*Z;i++)
    {
	cquad[i] = 0.2f;
	initial[i] = initial[i+X*Z] = 0.0f;
    }

    for(int s=0;s<n_source;s++)
    {
	for(int t=0;t<T;t++)
	{
	    if(t>100 && t<250)
		source[s*T + t] = 1.0f;
	    else
		source[s*T + t] = 0.0f;
	}
    }

    /////////////////////////////////////////////
    for(int i=0;i<50;i++)
    {
	cuda_simulate(X,Z,T,cquad,initial,en_out,n_source,pos_source_x,pos_source_z,source,n_sensor,pos_sensor_x,pos_sensor_z,recording);
    }
    free_mem_simulate();
    /////////////////////////////////////////////
    free(pos_source_x);
    free(pos_source_z);
    free(pos_sensor_x);
    free(pos_sensor_z);

    free(cquad);
    free(initial);
    free(source);
    free(recording);

    return 0;
}
