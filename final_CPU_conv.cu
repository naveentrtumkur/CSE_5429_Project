// Import the relevant header files.
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>


// While taking readings, modify N to small and large values to observe CPU compute_time
#define N 10000 //Default matrix size NxN

// Performing the Covolution operation at the host side.
void convolution_host(int A[][N+2], int C[][N+2])
{

	// Define the filter
	int filter[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

	// Now loop though the entire matrix performing the convolution
	for(int i=0;i<N+1;i++) //Looping through the rows..
	{
	    for(int j=0;j<N+1;j++) //Looping through the columns.
	    {
		if(0 < i && i < N + 1 && 0 < j && j < N + 1)
		{
		    // Triued using double pointer reference.
		    /* int value = 0; 
                     // Multiplying the matrix surrounding with the convolution filter.
                     value = value + *A((*((i - 1)*N)+ j - 1)) *  filter[0][0];
                     value = value + *A((i - 1)*N + j) *  filter[0][1];
                     value = value + *A((i - 1)*N j + 1) *  filter[0][2];
                     value = value + *A(i*N+ j - 1) *  filter[1][0];
                     value = value + *A(i*N+ j)   *  filter[1][1];
                     value = value + *A(i*N+ j + 1)   *  filter[1][2];
                     value = value + *A((i + 1)*N+ j - 1) *  filter[2][0];
                     value = value + *A((i + 1)*N+ j)  *  filter[2][1];
                     value = value + *A((i + 1)*N+ j + 1) *  filter[2][2];
                     C(i*N+ j) = value;*/

		     int value = 0;
		     value = value + A[i - 1] [j - 1] *  filter[0][0];
                     value = value + A[i - 1] [j] *  filter[0][1];
                     value = value + A[i - 1] [j + 1] *  filter[0][2];
                     value = value + A[i] [j - 1] *  filter[1][0];
                     value = value + A[i] [j]   *  filter[1][1];
                     value = value + A[i] [j + 1]   *  filter[1][2];
                     value = value + A[i + 1] [j - 1] *  filter[2][0];
                     value = value + A[i + 1][j]  *  filter[2][1];
                     value = value + A[i + 1] [j + 1] *  filter[2][2];
                     C[i] [j] = value;
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
	int C_h[N+2][N+2]={};	
	//Device variables
	int *A_d = 0, *C_d = 0;// A and C are variable used by kernel (GPU)

	//Needs for row-major layout
	int cols = N + 2;

	//Calculate memory size 
	int memorySize = (N + 2) * (N + 2);// Entire matrix need to be transferred, so we require N * N as size of gpu memory

	// timers to measure the appropriate time.
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Init matrix A and C_h to all 0 elements
	for (int i = 0; i < N+2; i++) {
		for (int j = 0; j < N+2; j++) {
			A[i][j] = 0;
			C_h[i][j] = 0;
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

	C = (int *)malloc(sizeof(*C)*memorySize);
	
	// perform the processing on the CPU
	clock_t begin = clock();
        //Conv2D_CPU(h_outImg_CPU, h_inImg, h_filter, imgWidth, imgHeight, imgChans);
        convolution_host(A, C_h); //Call teh host convolution method.
	clock_t end = clock();

        // calculate total time for CPU and GPU
        double time_execn = (double)(end - begin) / CLOCKS_PER_SEC*1000;
        printf("Total time for CPU execution is: %f milliseconds\n", time_execn);
	
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
	return EXIT_SUCCESS;
}

