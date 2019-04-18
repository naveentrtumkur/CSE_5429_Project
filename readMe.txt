############################################################
#######   Acceleration of Colvolution Operation using GPU ### 
###########################################################



Step #1: Make sure your have nvcc loaded and all GPU drivers setup.

Step #2: Parse and convert given image to grayscale and then to matrix of pixels
	 Use the python Parser for this.

Step #3: Compile the CPU and GPU code typing "make"

Step #4: TO execute CPU Code: $make CPU_Run

Step #5: To execute GPU code: $make GPU_Run

Step #6: To compare and contrast for different size of matrix, change the value of N in both CPU and GPu program
