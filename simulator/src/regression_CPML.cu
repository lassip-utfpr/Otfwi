#include <cuda_runtime_api.h>
#include <iostream>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "regression.h"

#define xzt(x,z,t) ((x) + (z)*(X) + ((t)%4)*(X)*(Z))
#define xzt2(x,z,t) ((x) + (z)*(X) + ((t)%4)*(X)*(Z))
#define inbounds(x,z,offset) (((x)>=(offset) && (z)>=(offset) && (x)<(X)-(offset) && (z)<(Z)-(offset)))

//precisam ser iguais
#define WARP_SIZE (32)
#define BLOCK_SIZE (32)

#define prec_deriv (4)
#include "deriv_macros.h"

typedef void (*adj_func)(float*, float*);


float *P, *cquad, *kappa_unrelaxed, *rho, *rho_half_x, *rho_half_z, *source, *record_buffer, *initial, *recording_h;
float *dx, *dz, *dt, *dt2, *qt_pi;
float *value_dpressure_dx, *value_dpressure_dz, *value_dpressurexx_dx, *value_dpressurezz_dz;
float *memory_dpressure_dx, *memory_dpressure_dz, *memory_dpressurexx_dx, *memory_dpressurezz_dz;
float *d_x, *K_x, *alpha_x, *a_x, *b_x, *d_x_half, *K_x_half, *alpha_x_half, *a_x_half, *b_x_half;
float *d_z, *K_z, *alpha_z, *a_z, *b_z, *d_z_half, *K_z_half, *alpha_z_half, *a_z_half, *b_z_half;
float *P_ub, *P_uf, *grad, *observed, *adj_source, *grad_h, *simulated_h, *adj_source_h;
float *P_uf_full;
int *pos_source_x, *pos_source_z, *pos_sensor_x, *pos_sensor_z;
int X, Z, T, n_source, n_sensor, pixel_per_element;
int allocated = 0;
unsigned int n_blocksX, n_blocksZ, n_blocksS, n_blocksF;

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

__global__ void
simulateFrame1 (float *P, 
    float *value_dpressure_dx, float *memory_dpressure_dx, 
    float *K_x_half, float *a_x_half, float *b_x_half, float *rho_half_x,
    float *dx,
    float *value_dpressure_dz, float *memory_dpressure_dz, 
    float *K_z_half, float *a_z_half, float *b_z_half, float *rho_half_z,
    float *dz,
    int X, int Z, int t)
{
    //coordenadas no bloco
    const int x = threadIdx.x;
    const int z = threadIdx.y;

    //coordenadas da origem do bloco
    const int x_b = blockIdx.x * blockDim.x;
    const int z_b = blockIdx.y * blockDim.y;

    //coordenadas em P (global)
    const int x_g = x_b + x;
    const int z_g = z_b + z;

    // coordenadas em Ps (shared)
    const int x_s = threadIdx.x + prec_deriv;
    const int z_s = threadIdx.y + prec_deriv;

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
                Ps[zz][xx] = P[xzt(x_c, z_c, t-1)];
            }  
            else
            {
                Ps[zz][xx] = 0.0f;
            }
        }
    }
    __syncthreads();
    
    const float dpressure_dx = derivPMLdx(Ps, 0, x_s, z_s);
    const float dpressure_dz = derivPMLdz(Ps, 0, x_s, z_s);

    if(inbounds(x_g, z_g, prec_deriv))
    {
        value_dpressure_dx[xzt(x_g,z_g,0)] =  dpressure_dx / *dx;
        memory_dpressure_dx[xzt(x_g,z_g,t)] = b_x_half[x_g] * memory_dpressure_dx[xzt(x_g,z_g,t-1)] + a_x_half[x_g] * value_dpressure_dx[xzt(x_g,z_g,0)];
        value_dpressure_dx[xzt(x_g,z_g,1)] = (value_dpressure_dx[xzt(x_g,z_g,0)] / K_x_half[x_g] + memory_dpressure_dx[xzt(x_g,z_g,t)]) / rho_half_x[xzt(x_g,z_g,0)];
        
        value_dpressure_dz[xzt(x_g,z_g,0)] = dpressure_dz / *dz;
        memory_dpressure_dz[xzt(x_g,z_g,t)] = b_z_half[z_g] * memory_dpressure_dz[xzt(x_g,z_g,t-1)] + a_z_half[z_g] * value_dpressure_dz[xzt(x_g,z_g,0)];
        value_dpressure_dz[xzt(x_g,z_g,1)] = (value_dpressure_dz[xzt(x_g,z_g,0)] / K_z_half[z_g] + memory_dpressure_dz[xzt(x_g,z_g,t)]) / rho_half_z[xzt(x_g,z_g,0)];
    }
}


__global__ void
simulateFrame2 (float *P, 
    float *value_dpressure_dx, float *value_dpressurexx_dx, float *memory_dpressurexx_dx, 
    float *K_x, float *a_x, float *b_x, float *dx,
    float *value_dpressure_dz, float *value_dpressurezz_dz, float *memory_dpressurezz_dz,
    float *K_z, float *a_z, float *b_z, float *dz, 
    float *dt2, float *kappa_unrelaxed,
    int X, int Z, int t)
{
    //coordenadas no bloco
    const int x = threadIdx.x;
    const int z = threadIdx.y;

    //coordenadas da origem do bloco
    const int x_b = blockIdx.x * blockDim.x;
    const int z_b = blockIdx.y * blockDim.y;

    //coordenadas em P (global)
    const int x_g = x_b + x;
    const int z_g = z_b + z;

    // coordenadas em Ps (shared)
    const int x_s = threadIdx.x + prec_deriv;
    const int z_s = threadIdx.y + prec_deriv;

    const int shared_width = BLOCK_SIZE + 2*prec_deriv;
    const int tam_shared = (shared_width)*(shared_width);

    //nao vale a pena colocar cquad e P(t-2) na memoria shared de acordo com os testes
    __shared__ float dPsdx[shared_width][shared_width];
    __shared__ float dPsdz[shared_width][shared_width];

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
                dPsdx[zz][xx] = value_dpressure_dx[xzt(x_c,z_c,1)];
                dPsdz[zz][xx] = value_dpressure_dz[xzt(x_c,z_c,1)];
            }
            else
            {
                dPsdx[zz][xx] = 0.0f;
                dPsdz[zz][xx] = 0.0f;
            }
        }
    }
    __syncthreads();

    const float dd_pressure_dxdx = derivPMLdx(dPsdx, 1, x_s, z_s);
    const float dd_pressure_dzdz = derivPMLdz(dPsdz, 1, x_s, z_s);

    if(inbounds(x_g, z_g, prec_deriv))
    {
        value_dpressurexx_dx[xzt(x_g,z_g,0)] = dd_pressure_dxdx / *dx;
        memory_dpressurexx_dx[xzt(x_g,z_g,t)] = b_x[x_g] * memory_dpressurexx_dx[xzt(x_g,z_g,t-1)] + a_x[x_g] * value_dpressurexx_dx[xzt(x_g,z_g,0)];
        value_dpressurexx_dx[xzt(x_g,z_g,1)] = value_dpressurexx_dx[xzt(x_g,z_g,0)] / K_x[x_g] + memory_dpressurexx_dx[xzt(x_g,z_g,t)];
        
        value_dpressurezz_dz[xzt(x_g,z_g,0)] = dd_pressure_dzdz / *dz;
        memory_dpressurezz_dz[xzt(x_g,z_g,t)] = b_z[z_g] * memory_dpressurezz_dz[xzt(x_g,z_g,t-1)] + a_z[z_g] * value_dpressurezz_dz[xzt(x_g,z_g,0)];
        value_dpressurezz_dz[xzt(x_g,z_g,1)] = value_dpressurezz_dz[xzt(x_g,z_g,0)] / K_z[z_g] + memory_dpressurezz_dz[xzt(x_g,z_g,t)];

        float lap = value_dpressurexx_dx[xzt(x_g,z_g,1)] + value_dpressurezz_dz[xzt(x_g,z_g,1)];

        P[xzt(x_g,z_g,t)] = 2 * P[xzt(x_g,z_g,t-1)] - P[xzt(x_g,z_g,t-2)] + lap * *dt2 * kappa_unrelaxed[xzt(x_g,z_g,0)];
    }
    else
    {
	    // apply Dirichlet conditions at the bottom of the C-PML layers
        // which is the right condition to implement in order for C-PML to remain stable at long times
        // Dirichlet condition
        // P[xzt(x_g,z_g,t)] = 0.0f;
    }
}


__global__ void
somaFonte(float *P, int X, int Z, int T, int t, int pixel_p_element, int n_fonte, int *pos_source_x, int *pos_source_z, float *dt2, float *qt_pi, float *cquad, float *source, int flip, int idx=-1)
{
    const int indexF = threadIdx.x + blockDim.x*threadIdx.y + blockIdx.x*(BLOCK_SIZE*BLOCK_SIZE);
    const int n = indexF; 

    if(pixel_p_element == 1)
    {
        if(n>=n_fonte || (idx!=-1 && idx!=n))
	    return;
    }
    else
    {
        if(n>=n_fonte)
        return;
    }
    

    float fonte;
    if(flip)
	fonte = source[n * T + T - 1 - t];
    else
	fonte = source[n*T + t];

    P[xzt(pos_source_x[n], pos_source_z[n], t)] += *qt_pi * cquad[xzt(pos_source_x[n], pos_source_z[n], 0)] * fonte * *dt2;
}


__global__ void
gravaBufferSensores2ordem(float *P, float *recording, int X, int Z, int T, int t, int *pos_sensor_x, int *pos_sensor_z, int n_sensor)
{
    //coordenadas no bloco
    const int indexS = threadIdx.x + blockDim.x*threadIdx.y + blockIdx.x*(BLOCK_SIZE*BLOCK_SIZE);
    const int n = indexS; 
    if(n<n_sensor)
	recording[n*T + t] = P[xzt(pos_sensor_x[n], pos_sensor_z[n], t)];
}


void
allocate_mem_simulate()
{
    cudaMalloc(&pos_source_x, n_source*sizeof(int)); //posicoes das fontes
    cudaMalloc(&pos_source_z, n_source*sizeof(int)); //posicoes das fontes
    cudaMalloc(&pos_sensor_x, n_sensor*sizeof(int)); //posicoes dos sensores
    cudaMalloc(&pos_sensor_z, n_sensor*sizeof(int)); //posicoes dos sensores 

    cudaMalloc(&dx, sizeof(float)); // discretização espacial x
    cudaMalloc(&dz, sizeof(float)); // discretização espacial z
    cudaMalloc(&dt, sizeof(float)); // discretização temporal
    cudaMalloc(&dt2, sizeof(float)); // dt ^ 2
    cudaMalloc(&qt_pi, sizeof(float)); // constante 4 * pi (soma da fonte)

    cudaMalloc(&cquad, X * Z * sizeof (float));	//campo de velocidades
    cudaMalloc(&kappa_unrelaxed, X * Z * sizeof(float)); 
    
    cudaMalloc(&rho, X * Z * sizeof(float)); // densidade
    cudaMalloc(&rho_half_x, X * Z * sizeof(float)); // densidade entre pontos x
    cudaMalloc(&rho_half_z, X * Z * sizeof(float)); // densidade entre pontos z

    cudaMalloc(&K_x, X * sizeof(float)); // K_x atenuação
    cudaMalloc(&a_x, X * sizeof(float)); // a_x atenuação
    cudaMalloc(&b_x, X * sizeof(float)); // b_x atenuação
    cudaMalloc(&K_x_half, X * sizeof(float)); // K_x_half atenuação
    cudaMalloc(&a_x_half, X * sizeof(float)); // a_x_half atenuação
    cudaMalloc(&b_x_half, X * sizeof(float)); // b_x_half atenuação

    cudaMalloc(&K_z, Z * sizeof(float)); // K_zz_g atenuação
    cudaMalloc(&a_z, Z * sizeof(float)); // a_z atenuação
    cudaMalloc(&b_z, Z * sizeof(float)); // b_z atenuação
    cudaMalloc(&K_z_half, Z * sizeof(float)); // K_z_half atenuação
    cudaMalloc(&a_z_half, Z * sizeof(float)); // a_z_half atenuação
    cudaMalloc(&b_z_half, Z * sizeof(float)); // b_z_half atenuação

    cudaMalloc(&source, T * n_source * sizeof (float));	//termos de fonte
    cudaMalloc(&initial, X * Z * 2 * sizeof (float));	//pressao direta
    cudaMalloc(&record_buffer, T * n_sensor * sizeof (float));	//buffer dos sensores
    cudaMallocHost(&recording_h, T*n_sensor*sizeof(float));
    
    // Pressão e Auxiliares
    cudaMalloc(&P, X * Z * 4 * sizeof (float));	//pressao direta
    cudaMalloc(&value_dpressure_dx, X * Z * 2 * sizeof(float)); 
    cudaMalloc(&value_dpressure_dz, X * Z * 2 * sizeof(float)); 
    cudaMalloc(&value_dpressurexx_dx, X * Z * 2 * sizeof(float)); 
    cudaMalloc(&value_dpressurezz_dz, X * Z * 2 * sizeof(float)); 
    cudaMalloc(&memory_dpressure_dx, X * Z * 4 * sizeof(float)); 
    cudaMalloc(&memory_dpressure_dz, X * Z * 4 * sizeof(float)); 
    cudaMalloc(&memory_dpressurexx_dx, X * Z * 4 * sizeof(float)); 
    cudaMalloc(&memory_dpressurezz_dz, X * Z * 4 * sizeof(float)); 
    
    allocated = 1;
}

void
free_mem_simulate()
{
    cudaFree (P);
    cudaFree (initial);
    cudaFree (cquad);
    cudaFree (source);
    cudaFree (record_buffer);

    cudaFree(kappa_unrelaxed);
    cudaFree(rho);
    cudaFree(rho_half_x);
    cudaFree(rho_half_z);

    cudaFree(K_x);
    cudaFree(a_x);
    cudaFree(b_x);
    cudaFree(K_x_half);
    cudaFree(a_x_half);
    cudaFree(b_x_half);

    cudaFree(K_z);
    cudaFree(a_z);
    cudaFree(b_z);
    cudaFree(K_z_half);
    cudaFree(a_z_half);
    cudaFree(b_z_half);

    cudaFree(pos_source_x);
    cudaFree(pos_source_z);
    cudaFree(pos_sensor_x);
    cudaFree(pos_sensor_z);

    cudaFree(value_dpressure_dx);
    cudaFree(value_dpressure_dz);
    cudaFree(value_dpressurexx_dx);
    cudaFree(value_dpressurezz_dz);

    cudaFree(memory_dpressure_dx);
    cudaFree(memory_dpressure_dz);
    cudaFree(memory_dpressurexx_dx);
    cudaFree(memory_dpressurezz_dz);

    cudaFree(recording_h);

    allocated = 0;
}

extern "C" void
init_memory_sim(int x, int z, int t, int ns, int nm, int ppe, 
    int *ps_x, int *ps_z, int *pm_x, int *pm_z,
    float dx_h, float dz_h, float dt_h, float dt2_h, float qtpi_h, 
    float *cq, float *kp_unr,
    float *rho_h, float *rho_hx_h, float *rho_hz_h,
    float *d_x_h, float *K_x_h, float *alpha_x_h, float *a_x_h, float *b_x_h,
    float *d_x_half_h, float *K_x_half_h, float *alpha_x_half_h, float *a_x_half_h, float *b_x_half_h,
    float *d_z_h, float *K_z_h, float *alpha_z_h, float *a_z_h, float *b_z_h,
    float *d_z_half_h, float *K_z_half_h, float *alpha_z_half_h, float *a_z_half_h, float *b_z_half_h,
	float *src, float *init, float **rec)
{
    // receber variáveis int primeiro, depois floats, por algum motivo dá 
    // errado receber int escalar depois de floats vetores
    X = x; 
    Z = z; 
    T = t; 
    n_source = ns; 
    n_sensor = nm;
    pixel_per_element = ppe;
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
    
    // re-ordering for clarity
    cudaMemcpy(pos_sensor_x, pm_x, n_sensor*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_sensor_z, pm_z, n_sensor*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_source_x, ps_x, n_source*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_source_z, ps_z, n_source*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(dx, &dx_h, 1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dz, &dz_h, 1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dt, &dt_h, 1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dt2, &dt2_h, 1*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(qt_pi, &qtpi_h, 1*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(cquad, cq, X * Z * sizeof (float), cudaMemcpyHostToDevice);
    cudaMemcpy(kappa_unrelaxed, kp_unr, X * Z * sizeof (float), cudaMemcpyHostToDevice);

    cudaMemcpy(rho, rho_h, X * Z * sizeof (float), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_half_x, rho_hx_h, X * Z * sizeof (float), cudaMemcpyHostToDevice);
    cudaMemcpy(rho_half_z, rho_hz_h, X * Z * sizeof (float), cudaMemcpyHostToDevice);

    cudaMemcpy(K_x, K_x_h, X * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(a_x, a_x_h, X * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_x, b_x_h, X * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(K_x_half, K_x_half_h, X * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(a_x_half, a_x_half_h, X * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_x_half, b_x_half_h, X * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(K_z, K_z_h, Z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(a_z, a_z_h, Z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_z, b_z_h, Z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(K_z_half, K_z_half_h, Z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(a_z_half, a_z_half_h, Z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_z_half, b_z_half_h, Z * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(source, src, T * n_source * sizeof (float), cudaMemcpyHostToDevice);
    cudaMemcpy(initial, init, X*Z*2*sizeof(float), cudaMemcpyHostToDevice);
}


extern "C" void
setCquad(float *cq)
{
    cudaMemcpy(cquad, cq, X * Z * sizeof (float), cudaMemcpyHostToDevice);
}


extern "C" void
set_source(int ns, int *ps_x, int *ps_z)
{
    n_source = ns;

    cudaMemcpy(pos_source_x, ps_x, n_source * sizeof (int), cudaMemcpyHostToDevice);
    cudaMemcpy(pos_source_z, ps_z, n_source * sizeof (int), cudaMemcpyHostToDevice);
}


extern "C" void
cuda_simulate2 (int en_out, int idx_source)
{
    FILE *pipeout;
    float *frame_buffer;
    if (en_out) 
    {
        char mpegCom[500];
        sprintf(mpegCom, "ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt gray -s %ix%i -r 60 -i - -f mp4 -q:v 5 -an -vcodec h264 -crf 0 output/output_CPML_s%i.mp4 -nostats -loglevel quiet", X, Z, idx_source);    
        pipeout = popen(mpegCom, "w");    
        cudaMallocHost(&frame_buffer, X*Z*sizeof(float));
    }

    //copia condicoes iniciais
    cudaMemset(P, 0, X*Z*4*sizeof(float));
    cudaMemcpy(P, initial, X*Z*2*sizeof(float), cudaMemcpyDeviceToDevice);
    
    cudaMemset(value_dpressure_dx, 0, X * Z * 2 * sizeof(float));
    cudaMemset(value_dpressure_dz, 0, X * Z * 2 * sizeof(float));
    cudaMemset(value_dpressurexx_dx, 0, X * Z * 2 * sizeof(float));
    cudaMemset(value_dpressurezz_dz, 0, X * Z * 2 * sizeof(float));
    cudaMemset(memory_dpressure_dx, 0, X * Z * 4 * sizeof(float));
    cudaMemset(memory_dpressure_dz, 0, X * Z * 4 * sizeof(float));
    cudaMemset(memory_dpressurexx_dx, 0, X * Z * 4 * sizeof(float));
    cudaMemset(memory_dpressurezz_dz, 0, X * Z * 4 * sizeof(float));
    



    for (int t = 0; t < T; t++)
    {
        // primeiros 2 frames sao condicao de contorno, logo nao calculados
        if (t > 1)
        {
            simulateFrame1 <<<blockGrid, threadGrid>>> (P, 
            value_dpressure_dx, memory_dpressure_dx,  
            K_x_half, a_x_half, b_x_half, rho_half_x, dx,
            value_dpressure_dz, memory_dpressure_dz, 
            K_z_half, a_z_half, b_z_half, rho_half_z, dz,
             X, Z, t);
            cudaDeviceSynchronize ();

            simulateFrame2 <<<blockGrid, threadGrid>>> (P, 
            value_dpressure_dx, value_dpressurexx_dx, memory_dpressurexx_dx, 
            K_x, a_x, b_x, dx,
            value_dpressure_dz, value_dpressurezz_dz, memory_dpressurezz_dz, 
            K_z, a_z, b_z, dz,
            dt2, kappa_unrelaxed,    
            X, Z, t);
            cudaDeviceSynchronize ();
                     
            somaFonte<<<n_blocksF, threadGrid>>>(P, X, Z, T, t, pixel_per_element, n_source, pos_source_x, pos_source_z, dt2, qt_pi, cquad, source, 0, idx_source);
            cudaDeviceSynchronize ();

        }

        // grava resultado nos sensores
        gravaBufferSensores2ordem<<<n_blocksS, threadGrid>>>(P, record_buffer, X, Z, T, t, pos_sensor_x, pos_sensor_z, n_sensor);
        cudaDeviceSynchronize();

        if (en_out)
        {
            cudaMemcpy(frame_buffer, &P[xzt(0,0,t)], X*Z*sizeof(float), cudaMemcpyDeviceToHost);
            writeFramePipe (pipeout, frame_buffer, X, Z, t, pos_sensor_x, pos_sensor_z, n_sensor);
        }

    }

    cudaMemcpy (recording_h, record_buffer, T * n_sensor * sizeof (float), cudaMemcpyDeviceToHost);

    if(en_out)
    {
        fflush(pipeout);
        pclose(pipeout);
        cudaFree(frame_buffer);
    }
}





