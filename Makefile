all: imgBlur

# PASCAL GT1080 and Volta
CUDA_ARCH=-gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 

imgBlur: imgBlur.o
	/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -m64 ${CUDA_ARCH} -o imgBlur imgBlur.o libwb/libwb.so -std=c++11

imgBlur.o: imgBlur.cu
	/usr/local/cuda-10.1/bin/nvcc -g -G -ccbin g++ -m64 ${CUDA_ARCH} -o imgBlur.o -c imgBlur.cu -std=c++11

#test input image size: 3840 x 2160
run: imgBlur
	./imgBlur -e golden_output.ppm -i input.ppm -o output.ppm -t image

debug: imgBlur
	cuda-gdb -tui --args ./imgBlur -e golden_output.ppm -i input.ppm -o output.ppm -t image

memcheck: imgBlur
	cuda-memcheck ./imgBlur -e golden_output.ppm -i input.ppm -o output.ppm -t image

clean:
	rm -rf *.o imgBlur output.ppm *~
