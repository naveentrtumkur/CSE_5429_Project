CC := nvcc

all:
	${CC} -o CPU_Conv final_CPU_conv.cu
	${CC} -o GPU_Conv final_GPU_conv.cu

clean:
	rm CPU_Conv GPU_Conv 

CPU_Run:
	./CPU_Conv

GPU_Run:
	./GPU_Conv	
