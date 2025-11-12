#pragma once

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

/* Utils */
__global__ void invert(unsigned char* data, int width, int height);

__global__ void bits_to_image(unsigned char* data, unsigned char* output_image, int width, int height, unsigned char bitmask);


/* Derivatives */
__global__ void fod(unsigned char* heightmap, int* fods, float* fod_dirs, int width, int height);

__global__ void watershed(int* fods, bool* watersheds, int width, int height);


/* Local maxima */
__global__ void non_maximum_suppression(unsigned char* heightmap, float* fod_dirs, bool* watershed, bool* suppressed, int width, int height);

__global__ void local_max_8dir(unsigned char* heightmap, unsigned char* local_max_8dirs, int width, int height);


/* Cone maps */
__global__ void create_cone_map_analytic(unsigned char* heightmap, bool* suppressed, float* fod_dirs, int* fods, unsigned char* cone_map, int width, int height);

__global__ void create_cone_map_8dir(unsigned char* heightmap, unsigned char* local_max_8dirs, unsigned char* cone_map, int width, int height);
