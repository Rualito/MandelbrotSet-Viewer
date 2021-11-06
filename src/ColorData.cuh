#include "Julia.cuh"
#include "cuErrorChecking.cuh"

//#define BENCHMARK
#include "BenchMark.hpp"

#define COUNT
#include "Debug.hpp"

#define THREADS_PER_BLOCK 16 // 1 dimensional
#define THREADS(N) dim3((THREADS_PER_BLOCK<N?THREADS_PER_BLOCK:N),1,1);
#define BLOCKS(N) dim3((N>THREADS_PER_BLOCK?N/THREADS_PER_BLOCK:1),1,1);


void getColorData(int gridN, double z0[2], double c0[2], double cstep[2], float** colorResult, int DEPTH){
    
    //gpuErrchk(cudaGetLastError(), false);


    dim3 blocks(gridN/THREADS_PER_BLOCK, gridN/THREADS_PER_BLOCK, 1);
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);

    double *d_c;
    double *d_z0;
    double *d_cstep;

    float *d_result;
    float *result = new float[gridN*gridN];

    // loading memory into device
    cudaMalloc( (void**)&d_c, 2*sizeof(double));
    cudaMalloc( (void**)&d_z0, 2*sizeof(double));
    cudaMalloc( (void**)&d_cstep, 2*sizeof(double));
    cudaMalloc( (void**)&d_result, gridN*gridN*sizeof(float));

    cudaMemcpy(d_c, c0, 2*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z0, z0, 2*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cstep, cstep, 2*sizeof(double), cudaMemcpyHostToDevice);

    // running threaded operations
    BENCHMARK_START(0);
    julia<<<blocks, threads>>>(d_c, d_z0, d_cstep,
			       DEPTH, gridN, d_result);
    //gpuErrchk(cudaGetLastError(), false);

    cudaDeviceSynchronize();
    BENCHMARK_END(0);
    
    // getting final results from cuda device
    cudaMemcpy(result, d_result, gridN*gridN*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0; i<gridN; ++i){
        for(int j=0; j<gridN; ++j){
            colorResult[i][j] = result[i*gridN+j];
            // printf("<%d, %d> %f\n", i, j, colorResult[i][j]);
        }
    }

    cudaFree(d_c);
    cudaFree(d_z0);
    cudaFree(d_cstep);
    cudaFree(d_result);
    delete[] result;

}
