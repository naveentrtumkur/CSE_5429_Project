// Import the relevant header files.
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>


// While taking readings, modify N to small and large values to observe GPU compute_time
#define N 10000 //Default matrix size NxN


// predefined functions to convert an element from pointer
#define A(i,j) A[(i)*cols+(j)]
#define C(i,j) C[(i)*cols+(j)]

// Kernel funciton to compute convolution on GPU.
__global__ void convolution(int *A, int *C)
{
	//Filter being used for convolution
	int filter[3][3] = { { 1, 2, 1 }, { 2, 4, 2 }, { 1, 2, 1 } };
	int cols = N +2; //Defien cols which would be used
	// Calculate the index on GPU block	
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Calculate the threads per block
	int tpb = (N+2)/ blockDim.x;//The amount of processing per thread

	for (int b = threadIdx.x * tpb; b < (threadIdx.x + 1) * tpb; b++){
		
		i = b;
		
		for (int j = 0; j < N + 1; j++){//columns iteration
			
			// Base condition to be checked to ignore zero padding
			if (0 < i && i < N + 1 && 0 < j && j < N + 1)
			{
				int val = 0;
				val = val + A(i - 1, j - 1) *  filter[0][0];
				val = val + A(i - 1, j) *  filter[0][1];
				val = val + A(i - 1, j + 1) *  filter[0][2];
				val = val + A(i, j - 1) *  filter[1][0];
				val = val + A(i, j) *  filter[1][1];
				val = val + A(i, j + 1) *  filter[1][2];
				val = val + A(i + 1, j - 1) *  filter[2][0];
				val = val + A(i + 1, j) *  filter[2][1];
				val = val + A(i + 1, j + 1) *  filter[2][2];
				C(i, j) = val;
			}
		}
	}

}

// Main Function
int main(void)
{
	//Host variables
	int A[N+2][N+2] = {};//+2 for padding matrix
	int *C;
	//int C_h[N+2][N+2]={};	
	//Device variables used on GPU side
	int *A_d = 0, *C_d = 0;// A and C are variable used by kernel (GPU)

	//Calculate the required memory size 
	int memSize = (N + 2) * (N + 2);// Entire matrix need to be transferred, so we require N * N as size of gpu memory

	// timers to measure the appropriate time.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Init matrix A and C_h to all 0 elements
	for (int i = 0; i < N+2; i++) {
		for (int j = 0; j < N+2; j++) {
			A[i][j] = 0;
			//C_h[i][j] = 0;
		}
	}

	//Generate random values between 0 and 9
	// Populating the matrix with random values between 0 and 9.
	srand(time(NULL));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A[i + 1][j + 1] = rand() % 10;
		}
	}

	C = (int *)malloc(sizeof(*C)*memSize);
	
	// allocate memory for device vars
	cudaMalloc((void**)&A_d, sizeof(*A_d)*memSize);
	cudaMalloc((void**)&C_d, sizeof(*C_d)*memSize);

	//Copy teh memory contents from host to device
	cudaMemcpy(A_d, A, sizeof(*A_d)*memSize, cudaMemcpyHostToDevice);

	// Start recording and measure the tim
	cudaEventRecord(start);
	convolution << <1, 512 >> >(A_d, C_d);//Block-thread
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//Copy from device to host
	cudaMemcpy(C, C_d, sizeof(*C)*memSize, cudaMemcpyDeviceToHost);


	/*
	// perform the processing on the CPU
	clock_t begin = clock();
        //Conv2D_CPU(h_outImg_CPU, h_inImg, h_filter, imgWidth, imgHeight, imgChans);
        convolution_host(A, C_h); //Call teh host convolution method.
	clock_t end = clock();

        // calculate total time for CPU and GPU
        double time_execn = (double)(end - begin) / CLOCKS_PER_SEC*1000;
        printf("Total time for CPU execution is: %f milliseconds\n", time_execn);
	*/
	
	// Verify that you're gettign the correct result for just a small matrix.
	////Print result
	/*printf("printing the result matrix");
	for (int i = 0; i < N + 2; i++) {
		for (int j = 0; j < N + 2; j++) {
			printf("%d ", C_h[i][j]);
		}
		printf("\n");
	}
	*/
	/*
	 //Print the original matrix
	printf("Printing the original matrix\n");
        for (int i = 0; i < N + 2; i++) {
                for (int j = 0; j < N + 2; j++) {
                        printf("%d ", A[i][j]);
                }
                printf("\n");
        }
	*/
	
	//Free up the allocated memory before exitiing
	cudaFree(C_d);
	cudaFree(A_d);
	free(C);
	
	// Report the execution time for processing
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken for GPU execution: %f", milliseconds);
	return 0;
}

