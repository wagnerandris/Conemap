#pragma once

// TODO is Sobel better than Prewitt?
__device__ __constant__ float vkernel[9] = {
 -1,-2,-1,
	0, 0, 0,
	1, 2, 1
};

__device__ __constant__ float hkernel[9] = {
 -1, 0, 1,
 -2, 0, 2,
 -1, 0, 1
};

__global__ void invert(unsigned char* data, int width, int height);

__global__ void local_max_8dirs(unsigned char* heightmap, unsigned char* dirs, int width, int height);

__global__ void bits_to_image(unsigned char* input, unsigned char* output_image, int width, int height, unsigned char bitmask);

__global__ void create_cone_map_8dirs(unsigned char* heightmap, unsigned char* derivative_image, unsigned char* dirs, unsigned char* cone_map, int width, int height);

__global__ void create_cone_map_4dirs(unsigned char* heightmap, unsigned char* derivative_image, unsigned char* dirs, unsigned char* cone_map, int width, int height);

__global__ void first_derivative(unsigned char* heightmap, unsigned char* fod_image, unsigned char* dirs, unsigned char* dirs_image, int* fod, int width, int height);

__global__ void second_derivative(int* fod, unsigned char* sod_image, unsigned char* watershed_image, int width, int height);

__global__ void non_maximum_suppression(unsigned char* heightmap, unsigned char* dirs, unsigned char* watershed_image, unsigned char* suppressed_image, int width, int height);

__global__ void create_cone_map_analytic(unsigned char* heightmap, unsigned char* fod_image, unsigned char* dirs, unsigned char* suppressed_image, unsigned char* cone_map, int width, int height);
