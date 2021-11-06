#ifndef H_ERRCHK_H
#define H_ERRCHK_H


//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define gpuErrchk(ans, abort) { gpuAssert((ans), __FILE__, __LINE__, abort); }


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif
