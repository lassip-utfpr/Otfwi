#include <iostream>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "regression.h"

#define xzt(x,z,t) ((x) + (z)*(X) + ((t)%2)*(X)*(Z))
#define inbounds(x,z,offset) (((x)>=(offset) && (z)>=(offset) && (x)<(X)-(offset) && (z)<(Z)-(offset)))

//precisam ser iguais
#define WARP_SIZE (32)
#define BLOCK_SIZE (32)

#define prec_deriv (4)
#include "deriv_macros.h"

typedef void (*adj_func)(float*, float*);

float *Ax, *Az, *Px, *Pz, *P, *d_x, *d_z, *dx, *dz, *dt, *cquad, *source, *record_buffer;
float *integral_source, *recording_h, *initial;
float *Ax_f, *Az_f, *Px_f, *Pz_f, *Ax_b, *Az_b, *Px_b, *Pz_b;
float *P_ub, *P_uf, *grad, *observed, *observed_h, *adj_source, *grad_h, *simulated_h;
float *adj_source_h, *integral_adjsource, *int_revsource;
int *pos_sensor_x, *pos_sensor_z, *pos_source_x, *pos_source_z; 
int *pos_revert_x, *pos_revert_z, n_revert;
float *rec_revert;
int X, Z, T, n_source, n_sensor;
int allocated = 0, allocated_reg = 0;
unsigned int n_blocksX, n_blocksZ, n_blocksS, n_blocksF, n_blocksR;

dim3 blockGrid;
const dim3 threadGrid(BLOCK_SIZE, BLOCK_SIZE);

typedef double take3double1int_givedouble (float *mse, float *observed, float *simulated, float *adj_source, int T,
					   int n_sensor);
take3double1int_givedouble *adj_calc;
extern "C" void
def_python_adj (take3double1int_givedouble * func)
{
    adj_calc = func;
}


void
allocate_mem_simulate()
{
    // aloca memória no device
    cudaMalloc(&Px, X*Z*2*sizeof(float)); //pressao direta x
    cudaMalloc(&Pz, X*Z*2*sizeof(float)); //pressao direta z
    cudaMalloc(&Ax, X*Z*2*sizeof(float)); //velocidade x
    cudaMalloc(&Az, X*Z*2*sizeof(float)); //velocidade z
    cudaMalloc(&dx, sizeof(float)); //discretização em x
    cudaMalloc(&dz, sizeof(float)); //discretização em z
    cudaMalloc(&dt, sizeof(float)); //discretização em t
    cudaMalloc(&P, X*Z*sizeof(float)); //pressao direta
    cudaMalloc(&d_x, X*Z*sizeof(float)); //atenuação x
    cudaMalloc(&d_z, X*Z*sizeof(float)); //atenuação x
    cudaMalloc(&cquad, X*Z*sizeof(float)); //campo de velocidades meio
    cudaMalloc(&source, T*n_source*sizeof(float)); //termos de fonte
    cudaMalloc(&pos_source_x, n_source*sizeof(int)); //posicoes das fontes
    cudaMalloc(&pos_source_z, n_source*sizeof(int)); //posicoes das fontes
    cudaMalloc(&pos_sensor_x, n_sensor*sizeof(int)); //posicoes dos sensores
    cudaMalloc(&pos_sensor_z, n_sensor*sizeof(int)); //posicoes dos sensores 
    cudaMalloc(&record_buffer, T*n_sensor*sizeof(float)); //buffer dos sensores
    cudaMalloc(&integral_source, n_source*sizeof(float)); //termos de fonte

    cudaMallocHost(&recording_h, T*n_sensor*sizeof(float));

    allocated = 1;
}

void
free_mem_simulate()
{
    cudaFree(pos_sensor_x);
    cudaFree(pos_sensor_z);
    cudaFree(pos_source_x);
    cudaFree(pos_source_z);
    cudaFree(integral_source);
    cudaFree(dx);
    cudaFree(dz);
    cudaFree(dt);
    cudaFree(P);
    cudaFree(Px);
    cudaFree(Pz);
    cudaFree(Ax);
    cudaFree(Az);
    cudaFree(d_x);
    cudaFree(d_z);
    cudaFree(record_buffer);
    cudaFree(cquad);
    cudaFree(source);

    cudaFree(recording_h);

    allocated = 0;
}


extern "C" void
init_memory_sim(int x, int z, int t, float *cq, float *const_vec, float *d_x_h, float *d_z_h, 
	int ns, int *ps_x, int *ps_z, int nm, int *pm_x, int *pm_z, 
	float *src, float *init, float **rec)
{
    X = x; 
    Z = z; 
    T = t; 
    n_source = ns; 
    n_sensor = nm;

    if(allocated)
	free_mem_simulate();

    cudaDeviceReset();

    allocate_mem_simulate();

    n_blocksX = X/BLOCK_SIZE;
    n_blocksZ = Z/BLOCK_SIZE;
    n_blocksS = n_sensor/(BLOCK_SIZE*BLOCK_SIZE);
    n_blocksF = n_source/(BLOCK_SIZE*BLOCK_SIZE);

    if(n_blocksX*BLOCK_SIZE != X)
	n_blocksX++;
    if(n_blocksZ*BLOCK_SIZE != Z)
	n_blocksZ++;
    if(n_blocksS*(BLOCK_SIZE*BLOCK_SIZE) != n_sensor) 
	n_blocksS++;
    if(n_blocksF*(BLOCK_SIZE*BLOCK_SIZE) != n_source) 
	n_blocksF++;


    blockGrid = {n_blocksX, n_blocksZ, 1};


    *rec = recording_h;
    cudaMalloc(&P, X * Z * 2 * sizeof (float));	//pressao direta
    cudaMemcpy(cquad, cq, X * Z * sizeof (float), cudaMemcpyHostToDevice);
    cudaMemcpy(source, src, T * n_source * sizeof (float), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_sensor_x, pm_x, n_sensor*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_sensor_z, pm_z, n_sensor*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_source_x, ps_x, n_source*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_source_z, ps_z, n_source*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(initial, init, X*Z*2*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dx, const_vec, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dz, const_vec+1, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dt, const_vec+2, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, d_x_h, X*Z*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, d_z_h, X*Z*sizeof(float), cudaMemcpyHostToDevice);
}

void
free_mem_regress()
{
    cudaFree(grad);
    cudaFree(observed);
    cudaFree(adj_source);
    cudaFree(integral_adjsource);
    cudaFree(int_revsource);

    cudaFree(P_ub);
    cudaFree(P_uf);
    cudaFree(Ax_f);
    cudaFree(Az_f);
    cudaFree(Px_f);
    cudaFree(Pz_f);
    cudaFree(Ax_b);
    cudaFree(Az_b);
    cudaFree(Px_b);
    cudaFree(Pz_b);

    cudaFree(simulated_h);
    cudaFree(adj_source_h);
    cudaFree(grad_h);

    cudaFree(pos_revert_x);
    cudaFree(pos_revert_z);
    cudaFree(rec_revert);

    cudaFree (P_ub);
    cudaFree (P_uf);
    cudaFree (observed);
    cudaFree (grad);
    cudaFree(adj_source);

    cudaFree(simulated_h);
    cudaFree(adj_source_h);
    cudaFree (grad_h);

    allocated_reg = 0;
}


void
allocate_mem_regress()
{
    cudaMalloc(&grad, X * Z * sizeof (float));	//gradiente final
    cudaMalloc(&observed, T * n_sensor * sizeof (float));	//buffer dos sensores
    cudaMalloc(&adj_source, T * n_sensor * sizeof (float));	//buffer do termo de fonte adjoint
    cudaMalloc(&integral_adjsource, n_sensor*sizeof(float)); //integral dos termos de fonte adjoint
    cudaMalloc(&int_revsource, n_source*sizeof(float)); //integral dos termos de fonte adjoint

    cudaMalloc(&P_ub, X * Z * 2 * sizeof (float));	//pressao adjoint backward
    cudaMalloc(&P_uf, X * Z * 2 * sizeof (float));	//pressao adjoint forward
    cudaMalloc(&Ax_f, X*Z*2*sizeof(float)); //velocidade x
    cudaMalloc(&Az_f, X*Z*2*sizeof(float)); //velocidade z
    cudaMalloc(&Px_f, X*Z*2*sizeof(float)); //pressao direta x
    cudaMalloc(&Pz_f, X*Z*2*sizeof(float)); //pressao direta z
    cudaMalloc(&Ax_b, X*Z*2*sizeof(float)); //velocidade x
    cudaMalloc(&Az_b, X*Z*2*sizeof(float)); //velocidade z
    cudaMalloc(&Px_b, X*Z*2*sizeof(float)); //pressao direta x
    cudaMalloc(&Pz_b, X*Z*2*sizeof(float)); //pressao direta z

    cudaMalloc(&pos_revert_x, n_revert*sizeof(int));
    cudaMalloc(&pos_revert_z, n_revert*sizeof(int));
    cudaMalloc(&rec_revert, T*n_revert*sizeof(float)); //pressao direta z

    cudaMallocHost(&simulated_h, T*n_sensor*sizeof(float));
    cudaMallocHost(&adj_source_h, T*n_sensor*sizeof(float));
    cudaMallocHost(&grad_h, X*Z*sizeof(float));

    allocated_reg = 1;
}


extern "C" void
init_memory_reg(float *obs, float **grad_ptr, int *p_rev_x, int *p_rev_z, int nr)
{
    
    n_revert = nr;

    if(allocated_reg)
	free_mem_regress();
    allocate_mem_regress();

    cudaMemcpy(pos_revert_x, p_rev_x, n_revert * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_revert_z, p_rev_z, n_revert * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(rec_revert, 0, T*n_revert*sizeof(float));


    n_blocksR = n_revert/(BLOCK_SIZE*BLOCK_SIZE);
    if(n_blocksR*(BLOCK_SIZE*BLOCK_SIZE) != n_revert) 
	n_blocksR++;

    cudaMemcpy(observed, obs, T * n_source * sizeof (float), cudaMemcpyHostToDevice);
    *grad_ptr = grad_h;
}


extern "C" void
setCquad(float *cq)
{
    cudaMemcpy(cquad, cq, X * Z * sizeof (float), cudaMemcpyHostToDevice);
}


__global__ void
setRec(float *Px, float *Pz, int X, int Z, int T, int t, int n_revert, int *pos_revert_x, int *pos_revert_z, float *revert)
{
    const int indexF = threadIdx.x + blockDim.x*threadIdx.y + blockIdx.x*(BLOCK_SIZE*BLOCK_SIZE);
    const int n = indexF; 

    if(n>=n_revert)
	return;

    Px[xzt(pos_revert_x[n], pos_revert_z[n], t)] = revert[n*T + T-1-t]/2;
    Pz[xzt(pos_revert_x[n], pos_revert_z[n], t)] = revert[n*T + T-1-t]/2;
    //Px[xzt(pos_revert_x[n], pos_revert_z[n], t)] = 0;
    //Pz[xzt(pos_revert_x[n], pos_revert_z[n], t)] = 0;
}


__global__ void
somaFonteIntegral(float *Px, float *Pz, int X, int Z, int T, int t, int n_fonte, int *pos_source_x, int *pos_source_z, float *source, float *integral, int flip, int idx=-1)
{
    const int indexF = threadIdx.x + blockDim.x*threadIdx.y + blockIdx.x*(BLOCK_SIZE*BLOCK_SIZE);
    const int n = indexF; 

    if(n>=n_fonte || (idx!=-1 && idx!=n))
	return;

    if(flip)
	integral[n] += source[n * T + T - 1 - t];
    else
	integral[n] += source[n*T + t];

    Px[xzt(pos_source_x[n], pos_source_z[n], t)] += integral[n]/2;
    Pz[xzt(pos_source_x[n], pos_source_z[n], t)] += integral[n]/2;
}


__global__ void
gravaBufferSensores(float *Px, float *Pz, float *buffer, int X, int Z, int T, int t, int n_sensors, int *pos_sensor_x, int *pos_sensor_z)
{
    const int indexS = threadIdx.x + blockDim.x*threadIdx.y + blockIdx.x*(BLOCK_SIZE*BLOCK_SIZE);
    const int n = indexS; 


    if(n<n_sensors)
    {
	buffer[n*T + t] = Px[xzt(pos_sensor_x[n], pos_sensor_z[n], t)] + Pz[xzt(pos_sensor_x[n], pos_sensor_z[n], t)];
    }
    //else
//	printf("index: %d\n", n);

}


__global__ void
somaFrames(float *destino, float *origem1, float *origem2, int X, int Z)
{
    const int x = threadIdx.x;
    const int z = threadIdx.y;

    //coordenadas da origem do bloco
    const int x_b = blockIdx.x * blockDim.x;
    const int z_b = blockIdx.y * blockDim.y;

    //coordenadas em P (global)
    const int x_g = x_b + x;
    const int z_g = z_b + z;

    destino[xzt(x_g,z_g,0)] = origem1[xzt(x_g,z_g,0)] + origem2[xzt(x_g,z_g,0)];
}

__global__ void
calcGrad(float *Px, float *Pz, float *Pu_f, float *grad, int X, int Z, int T, int t)
{
    //coordenadas no bloco
    const int x = threadIdx.x;
    const int z = threadIdx.y;

    //coordenadas em Ps (shared)
    const int x_s = threadIdx.x + prec_deriv;
    const int z_s = threadIdx.y + prec_deriv;

    //coordenadas da origem do bloco
    const int x_b = blockIdx.x * blockDim.x;
    const int z_b = blockIdx.y * blockDim.y;

    //coordenadas em P (global)
    const int x_g = x_b + x;
    const int z_g = z_b + z;

    const int shared_width = BLOCK_SIZE + 2*prec_deriv;
    const int tam_shared = (shared_width)*(shared_width);

    //nao vale a pena colocar cquad e P(t-2) na memoria shared de acordo com os testes
    __shared__ float Psx[shared_width][shared_width];
    __shared__ float Psz[shared_width][shared_width];

    if(z==0) //first warp in block
    {
	//copy P to shared memory
	for(int id=x; id<tam_shared; id+=WARP_SIZE)
	{
	    //coordenada dentro de Ps (shared) sendo lida de P
	    const int xx = id/shared_width;
	    const int zz = id%shared_width;

	    //coordenada correspondente na memoria global
	    const int x_c = x_b - prec_deriv + xx;
	    const int z_c = z_b - prec_deriv + zz;

	    //retirar if adicionando o anel de zeros
	    if(inbounds(x_c, z_c, prec_deriv))
	    {
		Psx[zz][xx] = Px[xzt(x_c, z_c, t)];
		Psz[zz][xx] = Pz[xzt(x_c, z_c, t)];
	    }
	    else
	    {
		Psx[zz][xx] = 0.0f;
		Psz[zz][xx] = 0.0f;
	    }
	}
    }
    __syncthreads();

    float lapP = deriv_x(Psx, x_s, z_s) + deriv_z(Psx, x_s, z_s);
    lapP += deriv_x(Psz, x_s, z_s) + deriv_z(Psz, x_s, z_s);
    grad[xzt(x_g, z_g, 0)] += Pu_f[xzt(x_g,z_g,0)] * lapP;
    //grad[xzt(x_g, z_g, 0)] += Pu_f[xzt(x_g,z_g,0) + X*Z*(T-1-t)] * lapP;
}


__global__ void
calcGrad2(float *P_a, float *P, float *grad, int X, int Z, int T, int t)
{
    //coordenadas no bloco
    const int x = threadIdx.x;
    const int z = threadIdx.y;

    //coordenadas em Ps (shared)
    const int x_s = threadIdx.x + prec_deriv;
    const int z_s = threadIdx.y + prec_deriv;

    //coordenadas da origem do bloco
    const int x_b = blockIdx.x * blockDim.x;
    const int z_b = blockIdx.y * blockDim.y;

    //coordenadas em P (global)
    const int x_g = x_b + x;
    const int z_g = z_b + z;

    const int shared_width = BLOCK_SIZE + 2*prec_deriv;
    const int tam_shared = (shared_width)*(shared_width);

    //nao vale a pena colocar cquad e P(t-2) na memoria shared de acordo com os testes
    __shared__ float Ps[shared_width][shared_width];

    if(z==0) //first warp in block
    {
	//copy P to shared memory
	for(int id=x; id<tam_shared; id+=WARP_SIZE)
	{
	    //coordenada dentro de Ps (shared) sendo lida de P
	    const int xx = id/shared_width;
	    const int zz = id%shared_width;

	    //coordenada correspondente na memoria global
	    const int x_c = x_b - prec_deriv + xx;
	    const int z_c = z_b - prec_deriv + zz;

	    //retirar if adicionando o anel de zeros
	    if(inbounds(x_c, z_c, prec_deriv))
	    {
		Ps[zz][xx] = P[xzt(x_c, z_c, 0)];
	    }
	    else
	    {
		Ps[zz][xx] = 0.0f;
		Ps[zz][xx] = 0.0f;
	    }
	}
    }
    __syncthreads();

    float lapP = deriv_x(Ps, x_s, z_s) + deriv_z(Ps, x_s, z_s);
    grad[xzt(x_g, z_g, 0)] += P_a[xzt(x_g,z_g,0)] * lapP;
    //grad[xzt(x_g, z_g, 0)] += Pu_f[xzt(x_g,z_g,0) + X*Z*(T-1-t)] * lapP;
}

__global__
void simulateFrameP(float *Px, float *Pz, float *Ax, float *Az,
			float *d_x, float *d_z, float *cquad, float *dx, float *dz, float *dt,
			int t, int X, int Z, int revert=0)
{
    //coordenadas no bloco
    const int x = threadIdx.x;
    const int z = threadIdx.y;

    //coordenadas em Ps (shared)
    const int x_s = threadIdx.x + prec_deriv;
    const int z_s = threadIdx.y + prec_deriv;

    //coordenadas da origem do bloco
    const int x_b = blockIdx.x * blockDim.x;
    const int z_b = blockIdx.y * blockDim.y;

    //coordenadas em P (global)
    const int x_g = x_b + x;
    const int z_g = z_b + z;

    const int shared_width = BLOCK_SIZE + 2*prec_deriv;
    const int tam_shared = (shared_width)*(shared_width);

    //nao vale a pena colocar cquad e P(t-2) na memoria shared de acordo com os testes
    __shared__ float Asx[shared_width][shared_width];
    __shared__ float Asz[shared_width][shared_width];

    if(z==0) //first warp in block
    {
	//copy P to shared memory
	for(int id=x; id<tam_shared; id+=WARP_SIZE)
	{
	    //coordenada dentro de Ps (shared) sendo lida de P
	    const int xx = id/shared_width;
	    const int zz = id%shared_width;

	    //coordenada correspondente na memoria global
	    const int x_c = x_b - prec_deriv + xx;
	    const int z_c = z_b - prec_deriv + zz;

	    //retirar if adicionando o anel de zeros
	    if(inbounds(x_c, z_c, prec_deriv))
	    {
		Asx[zz][xx] = Ax[xzt(x_c, z_c, t-1)];
		Asz[zz][xx] = Az[xzt(x_c, z_c, t-1)];
	    }
	    else
	    {
		Asx[zz][xx] = 0.0f;
		Asz[zz][xx] = 0.0f;
	    }
	}
    }
    __syncthreads();
    
    const float dAxdx = derivPMLdx(Asx, 1, x_s, z_s);
    const float dAzdz = derivPMLdz(Asz, 1, x_s, z_s);
    const float d_xx = (d_x[xzt(x_g+1,z_g,0)]);
    const float d_zz = (d_z[xzt(x_g,z_g+1,0)]);

    if(inbounds(x_g, z_g, prec_deriv))
    {
	if(!revert)
	{
	    Px[xzt(x_g,z_g,t)] = Px[xzt(x_g,z_g,t-1)] * (1 - d_xx * (*dt)) + cquad[xzt(x_g,z_g,0)] * *dt/(*dx) * dAxdx;
	    Pz[xzt(x_g,z_g,t)] = Pz[xzt(x_g,z_g,t-1)] * (1 - d_zz * (*dt)) + cquad[xzt(x_g,z_g,0)] * *dt/(*dx) * dAzdz;
	}
	else
	{
	    Px[xzt(x_g,z_g,t)] = (Px[xzt(x_g,z_g,t-1)] - cquad[xzt(x_g,z_g,0)] * (*dt/(*dx)) * dAxdx);// /(1 - d_xx*(*dt));
	    Pz[xzt(x_g,z_g,t)] = (Pz[xzt(x_g,z_g,t-1)] - cquad[xzt(x_g,z_g,0)] * (*dt/(*dx)) * dAzdz);// /(1 - d_zz*(*dt));
	}
    }
    else
    {
	//Px[xzt(x_g,z_g,t)] = 0.0f;
	//Pz[xzt(x_g,z_g,t)] = 0.0f;
    }


}


__global__
void simulateFrameA(float *Px, float *Pz, float *Ax, float *Az,
		    	float *d_x, float *d_z, float *cquad, float *dx, float *dz, float *dt,
			int t, int X, int Z, int revert=0)
{
    //coordenadas no bloco
    const int x = threadIdx.x;
    const int z = threadIdx.y;

    //coordenadas em Ps (shared)
    const int x_s = threadIdx.x + prec_deriv;
    const int z_s = threadIdx.y + prec_deriv;

    //coordenadas da origem do bloco
    const int x_b = blockIdx.x * blockDim.x;
    const int z_b = blockIdx.y * blockDim.y;

    //coordenadas em P (global)
    const int x_g = x_b + x;
    const int z_g = z_b + z;

    const int shared_width = BLOCK_SIZE + 2*prec_deriv;
    const int tam_shared = (shared_width)*(shared_width);

    //nao vale a pena colocar cquad e P(t-2) na memoria shared de acordo com os testes
    __shared__ float Psx[shared_width][shared_width];
    __shared__ float Psz[shared_width][shared_width];

    if(z==0) //first warp in block
    {
	//copy P to shared memory
	for(int id=x; id<tam_shared; id+=WARP_SIZE)
	{
	    //coordenada dentro de Ps (shared) sendo lida de P
	    const int xx = id/shared_width;
	    const int zz = id%shared_width;

	    //coordenada correspondente na memoria global
	    const int x_c = x_b - prec_deriv + xx;
	    const int z_c = z_b - prec_deriv + zz;

	    //retirar if adicionando o anel de zeros
	    if(inbounds(x_c, z_c, prec_deriv))
	    {
		Psx[zz][xx] = Px[xzt(x_c, z_c, t)];
		Psz[zz][xx] = Pz[xzt(x_c, z_c, t)];
	    }
	    else
	    {
		Psx[zz][xx] = 0.0f;
		Psz[zz][xx] = 0.0f;
	    }
	}
    }
    __syncthreads();
    
    const float dPxdx = derivPMLdx(Psx, 0, x_s, z_s);
    const float dPxdz = derivPMLdz(Psx, 0, x_s, z_s);
    const float dPzdx = derivPMLdx(Psz, 0, x_s, z_s);
    const float dPzdz = derivPMLdz(Psz, 0, x_s, z_s);

    const float d_xx = (d_x[xzt(x_g+1,z_g,0)]);
    const float d_zz = (d_z[xzt(x_g,z_g+1,0)]);


    //assume que nao tem anel de zeros forcados por indice, entao usa as bordas como anel
    if(inbounds(x_g, z_g, prec_deriv))
    {
	if(!revert)
	{
	    Ax[xzt(x_g,z_g,t)] = Ax[xzt(x_g,z_g,t-1)] * (1 - d_xx*(*dt)) + *dt/(*dx) * (dPxdx + dPzdx);
	    Az[xzt(x_g,z_g,t)] = Az[xzt(x_g,z_g,t-1)] * (1 - d_zz*(*dt)) + *dt/(*dx) * (dPxdz + dPzdz);
	}
	else
	{
	    Ax[xzt(x_g,z_g,t)] = (Ax[xzt(x_g,z_g,t-1)] - *dt/(*dx) * (dPxdx + dPzdx));// /(1 - d_xx * *dt);
	    Az[xzt(x_g,z_g,t)] = (Az[xzt(x_g,z_g,t-1)] - *dt/(*dx) * (dPxdz + dPzdz));// /(1 - d_zz * *dt);
	}
    }
    else
    {
	//Ax[xzt(x_g,z_g,t)] = 0.0f;
        //Az[xzt(x_g,z_g,t)] = 0.0f;
    }
}


extern "C"
void cuda_simulate(int en_out, int idx_source)
{
    FILE *pipeout;
    float *frame_buffer;
    if(en_out)
    {
	char mpegCom[500];
	sprintf(mpegCom, "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt gray -s %ix%i -r 20 -i - -f mp4 -q:v 5 -an -vcodec h264 -crf 0 output/outputPML.mp4 -nostats -loglevel quiet", Z, X);    
	pipeout = popen(mpegCom, "w");    
	cudaMallocHost(&frame_buffer, X*Z*sizeof(float));
    }



    // limpa diretorios de saída
    int saida = 0;
    if(en_out)
    {
        saida += system("mkdir -p output/images");
	saida += system("rm output/images/*.blob -f");
    }
    if(saida)
    {
    	printf("Erro limpando diretórios!");
    }

    //zera matrizes
    cudaMemset(Px, 0, X*Z*2*sizeof(float));
    cudaMemset(Pz, 0, X*Z*2*sizeof(float));
    cudaMemset(Ax, 0, X*Z*2*sizeof(float));
    cudaMemset(Az, 0, X*Z*2*sizeof(float));
    cudaMemset(integral_source, 0, n_source*sizeof(float));

    // copia condicoes iniciais
    cudaMemcpy(Px, initial, X*Z*2*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(Pz, initial, X*Z*2*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(Ax, initial, X*Z*2*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(Az, initial, X*Z*2*sizeof(float), cudaMemcpyDeviceToDevice);


    // simulacao da propagação
    // primeiro frame é condicao de contorno, logo nao calculado
    for(int t = 1; t<T; t++)
    {
	// Atualiza as pressões
	simulateFrameP<<<blockGrid, threadGrid>>>(Px, Pz, Ax, Az, d_x, d_z, cquad, dx, dz, dt, t, X, Z);
	cudaDeviceSynchronize();

	//soma termos de fonte
	somaFonteIntegral<<<n_blocksF, threadGrid>>>(Px, Pz, X, Z, T, t, n_source, pos_source_x, pos_source_z, source, integral_source, 0, idx_source);
	cudaDeviceSynchronize();

	// grava resultado nos sensores
	gravaBufferSensores<<<n_blocksS, threadGrid>>>(Px, Pz, record_buffer, X, Z, T, t, n_sensor, pos_sensor_x, pos_sensor_z);
	cudaDeviceSynchronize();

	// Atualiza as velocidades
	simulateFrameA<<<blockGrid, threadGrid>>>(Px, Pz, Ax, Az, d_x, d_z, cquad, dx, dz, dt, t, X, Z);
	cudaDeviceSynchronize();
		
	
	// gera arquivos de saída
	if(en_out)
	{
	    // soma Px e Pz para exibição
	    somaFrames<<<blockGrid, threadGrid>>>(P, &Px[xzt(0,0,t)], &Pz[xzt(0,0,t)], X, Z);
	    cudaMemcpy(frame_buffer, P, X*Z*sizeof(float), cudaMemcpyDeviceToHost);
	    writeFramePipe (pipeout, frame_buffer, X, Z, t, pos_sensor_x, pos_sensor_z, n_sensor);
	    //writeFrame (P, X, Z, t, pos_sensor_x, pos_sensor_z, n_sensor);
        }
    }

    
    // copia dados do device pro host
    cudaMemcpy (recording_h, record_buffer, T * n_sensor * sizeof (float), cudaMemcpyDeviceToHost);

    // fecha pipe do video
    if(en_out)
    {
	fflush(pipeout);
	pclose(pipeout);
	cudaFree(frame_buffer);
    }
}

extern "C" void
cuda_grad_ext(float *mse_h, adj_func adj_calc, int idx_source)
{
    float mse = 0.0f;

    cudaMemset(rec_revert, 0, T*n_revert*sizeof(float));
    float *Pu_b;
    cudaMalloc(&Pu_b, sizeof(float)*X*Z);

    //float *full_P = (float*)malloc(X*Z*T*sizeof(float));
    //float *full_Pb = (float*)malloc(X*Z*T*sizeof(float));
    //float *P_uf_full;
    //cudaMalloc(&P_uf_full, X*Z*T*sizeof(float)); //campo adjoint completo

    //FILE *pipeout;
    //char mpegCom[500];
    //float *frame_buffer;
    //cudaMallocHost(&frame_buffer, X*Z*sizeof(float));

    //zera campos de pressao
    cudaMemset(P, 0, X*Z*sizeof(float));
    cudaMemset(P_ub, 0, X*Z*2*sizeof(float));
    cudaMemset(P_uf, 0, X*Z*2*sizeof(float));
    cudaMemset(grad, 0, X*Z*sizeof(float));
    cudaMemset(Px, 0, X*Z*2*sizeof(float));
    cudaMemset(Pz, 0, X*Z*2*sizeof(float));
    cudaMemset(Ax, 0, X*Z*2*sizeof(float));
    cudaMemset(Az, 0, X*Z*2*sizeof(float));
    cudaMemset(Px_f, 0, X*Z*2*sizeof(float));
    cudaMemset(Pz_f, 0, X*Z*2*sizeof(float));
    cudaMemset(Ax_f, 0, X*Z*2*sizeof(float));
    cudaMemset(Az_f, 0, X*Z*2*sizeof(float));
    cudaMemset(Px_b, 0, X*Z*2*sizeof(float));
    cudaMemset(Pz_b, 0, X*Z*2*sizeof(float));
    cudaMemset(Ax_b, 0, X*Z*2*sizeof(float));
    cudaMemset(Az_b, 0, X*Z*2*sizeof(float));

    cudaMemset(integral_source, 0, n_source*sizeof(float));
    cudaMemset(integral_adjsource, 0, n_sensor*sizeof(float));

    //FASE 1 - SIMULAÇÃO DIRETA    
    for (int t = 0; t < T; t++)
    {
	if(t>0)
	{
	// Atualiza as pressões
	simulateFrameP<<<blockGrid, threadGrid>>>(Px, Pz, Ax, Az, d_x, d_z, cquad, dx, dz, dt, t, X, Z);
	cudaDeviceSynchronize();

	//soma termos de fonte
	somaFonteIntegral<<<n_blocksF, threadGrid>>>(Px, Pz, X, Z, T, t, n_source, pos_source_x, pos_source_z, source, integral_source, 0, idx_source);
	cudaDeviceSynchronize();

	// grava resultado nos sensores
	gravaBufferSensores<<<n_blocksR, threadGrid>>>(Px, Pz, rec_revert, X, Z, T, t, n_revert, pos_revert_x, pos_revert_z);
	gravaBufferSensores<<<n_blocksS, threadGrid>>>(Px, Pz, record_buffer, X, Z, T, t, n_sensor, pos_sensor_x, pos_sensor_z);
	cudaDeviceSynchronize();

	// Atualiza as velocidades
	simulateFrameA<<<blockGrid, threadGrid>>>(Px, Pz, Ax, Az, d_x, d_z, cquad, dx, dz, dt, t, X, Z);
	cudaDeviceSynchronize();
	}
	//somaFrames<<<blockGrid, threadGrid>>>(P, &Px[xzt(0,0,t)], &Pz[xzt(0,0,t)], X, Z);
	//cudaDeviceSynchronize();
	//cudaMemcpy(&full_P[X*Z*t], P, X*Z*sizeof(float), cudaMemcpyDeviceToHost);
    }

    //armazena campo adjoint em todos os instantes para uso na fase 3
    //somaFrames<<<blockGrid, threadGrid>>>(P, &Px[xzt(0,0,t)], &Pz[xzt(0,0,t)], X, Z);
    //cudaDeviceSynchronize();
    //cudaMemcpy(&full_P[X*Z*t], P, X*Z*sizeof(float), cudaMemcpyDeviceToHost);


    //FASE 1A - CÁLCULO DO MSE E ADJOINT SOURCE
    cudaMemcpy(simulated_h, record_buffer, T*n_sensor*sizeof(float), cudaMemcpyDeviceToHost);
    adj_calc (simulated_h, adj_source_h);
    cudaMemcpy(adj_source, adj_source_h, T*n_sensor*sizeof(float), cudaMemcpyHostToDevice);

    //FASE 2 - SIMULAÇÃO ADJOINT FORWARD

	//sprintf(mpegCom, "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt gray -s %ix%i -r 20 -i - -f mp4 -q:v 5 -an -vcodec h264 -crf 0 output/output_adj.mp4 -nostats -loglevel quiet", Z, X);    
	//pipeout = popen(mpegCom, "w");    

    cudaMemcpy(Px_b, &Px[xzt(0,0,T-1)],  X*Z*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(Pz_b, &Pz[xzt(0,0,T-1)],  X*Z*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(Ax_b, &Ax[xzt(0,0,T-2)],  X*Z*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(Az_b, &Az[xzt(0,0,T-2)],  X*Z*sizeof(float), cudaMemcpyDeviceToDevice);


    for (int t = 0; t < T; t++)
    {
	if(t>0)
	{
	    // Atualiza as pressões
	    simulateFrameP<<<blockGrid, threadGrid>>>(Px_f, Pz_f, Ax_f, Az_f, d_x, d_z, cquad, dx, dz, dt, t, X, Z);
	    simulateFrameP<<<blockGrid, threadGrid>>>(Px_b, Pz_b, Ax_b, Az_b, d_x, d_z, cquad, dx, dz, dt, t, X, Z, 1);
	    cudaDeviceSynchronize();

	    //soma termo de fonte adjoint
	    somaFonteIntegral<<<n_blocksS, threadGrid>>>(Px_f, Pz_f, X, Z, T, t, n_sensor, pos_sensor_x, pos_sensor_z, adj_source, integral_adjsource, 1);
	    // seta condicoes de contorno da reversao
	    setRec<<<n_blocksR, threadGrid>>>(Px_b, Pz_b, X, Z, T, t, n_revert, pos_revert_x, pos_revert_z, rec_revert);
	    cudaDeviceSynchronize();

	    // Atualiza as velocidades
	    simulateFrameA<<<blockGrid, threadGrid>>>(Px_f, Pz_f, Ax_f, Az_f, d_x, d_z, cquad, dx, dz, dt, t, X, Z);
	    simulateFrameA<<<blockGrid, threadGrid>>>(Px_b, Pz_b, Ax_b, Az_b, d_x, d_z, cquad, dx, dz, dt, t, X, Z, 1);
	    cudaDeviceSynchronize();
	}

	// TODO: retirar somaframes e passar os 4 campos pro calcgrad
	somaFrames<<<blockGrid, threadGrid>>>(Pu_b, &Px_b[xzt(0,0,t)], &Pz_b[xzt(0,0,t)], X, Z);
	somaFrames<<<blockGrid, threadGrid>>>(P, &Px_f[xzt(0,0,t)], &Pz_f[xzt(0,0,t)], X, Z);
	cudaDeviceSynchronize();

	//cudaMemcpy(&full_Pb[X*Z*(T-1-t)], Pu_b, X*Z*sizeof(float), cudaMemcpyDeviceToHost);
	// TODO: test gate
	//if(t>1000)
	{
	    calcGrad2<<<blockGrid, threadGrid>>>(P, Pu_b, grad, X, Z, T, t);
	    cudaDeviceSynchronize();
	}

	//cudaMemcpy(&full_Pb[X*Z*t], Pu_b, X*Z*sizeof(float), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&full_P[X*Z*t], P, X*Z*sizeof(float), cudaMemcpyDeviceToHost);

	    //cudaMemcpy(frame_buffer, &P_uf_full[X*Z*(T-1-t)], X*Z*sizeof(float), cudaMemcpyDeviceToHost);
	    //cudaMemcpy(frame_buffer, Pu_b, X*Z*sizeof(float), cudaMemcpyDeviceToHost);
	    //cudaMemcpy(frame_buffer, P, X*Z*sizeof(float), cudaMemcpyDeviceToHost);
	    //writeFramePipe (pipeout, frame_buffer, X, Z, t, pos_sensor_x, pos_sensor_z, n_sensor);
    }
	//fflush(pipeout);
	//pclose(pipeout);

    //adj_calc (full_P, full_Pb);
    cudaMemcpy (grad_h, grad, X * Z * sizeof (float), cudaMemcpyDeviceToHost);
    *mse_h = mse;


    //cudaFree (P_uf_full);
    //cudaFree(frame_buffer);
    cudaFree(Pu_b);
}


