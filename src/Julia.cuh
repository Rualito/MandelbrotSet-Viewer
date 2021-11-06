
#ifndef JULIA_H
#define JULIA_H

__global__
void julia(double c[2], double z0[2], double cstep[2], 
	   int MAX_ITER, int n, float* result){
    const int index_x = blockIdx.x*blockDim.x+threadIdx.x;
    const int index_y = blockIdx.y*blockDim.y+threadIdx.y;
    
    if(index_x<n && index_y<n){
        const double cnow[2]{c[0]+index_x*cstep[0],
		             c[1]+index_y*cstep[1]};
		
		//
        double temp[2]{z0[0], z0[1]};
		
		// in case nothing happens
		result[index_x*n+index_y] = 1.;
		
		for(int i=0; i<MAX_ITER; ++i){
			double temp0 = temp[0];
			temp[0] = cnow[0] + (temp[0]*temp[0]-temp[1]*temp[1]);
			temp[1] = cnow[1] + 2*temp[1]*temp0;

			if(temp[0]*temp[0]+temp[1]*temp[1]>4){
				result[index_x*n+index_y] = (i+0.)/MAX_ITER;
				break;
			}
		}
    }
}

#endif
