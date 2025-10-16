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

/* Utils */
__global__ void invert(unsigned char* data, int width, int height);

__global__ void bits_to_image(unsigned char* data, unsigned char* output_image, int width, int height, unsigned char bitmask);


/* Derivatives */
__global__ void fod(unsigned char* heightmap, int* fod, unsigned char* fod_image, float* exact_dirs, unsigned char* discrete_dirs, unsigned char* dirs_image, int width, int height);

__global__ void sod_and_watershed(int* fod, unsigned char* sod_image, unsigned char* watershed_image, int width, int height);


/* Local maxima */
__global__ void non_maximum_suppression(unsigned char* heightmap, unsigned char* dirs, unsigned char* watershed_image, unsigned char* suppressed_image, int width, int height);

__global__ void local_max_8dir(unsigned char* heightmap, unsigned char* local_max_8dirs, int width, int height);

__global__ void create_binary_mipmap_level(unsigned char* input, int width, int height, unsigned char* binary_mipmap, int mipmap_width, int mipmap_height);

__global__ void create_max_mipmap_level(unsigned char* input, int width, int height, unsigned char* max_mipmap, int mipmap_width, int mipmap_height);

// TODO template function???
/* Cone maps */
__global__ void create_cone_map_baseline(unsigned char* heightmap, unsigned char* fod_image, float* gradient_dirs, unsigned char* watershed, unsigned char* cone_map, int width, int height);

__global__ void create_cone_map_analytic(unsigned char* heightmap, unsigned char* fod_image, unsigned char* dirs, unsigned char* suppressed_image, unsigned char* cone_map, int width, int height);

__global__ void create_cone_map_8dir(unsigned char* heightmap, unsigned char* fod_image, unsigned char* local_max_8dirs, unsigned char* cone_map, int width, int height);

__global__ void create_cone_map_4dir(unsigned char* heightmap, unsigned char* fod_image, unsigned char* local_max_8dirs, unsigned char* cone_map, int width, int height);
