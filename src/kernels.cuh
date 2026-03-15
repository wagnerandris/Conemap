#pragma once


/* Utils */
__global__ void pack(uint8_t* __restrict__ heightmap, uint16_t* __restrict__ packed, int width, int height);

__global__ void pack_continuously(const uint8_t* __restrict__ heightmap, uint16_t* __restrict__ packed, int width, int height);

__global__ void invert(uint8_t* __restrict__ data, int width, int height);

__global__ void bits_to_image(uint8_t* __restrict__ data, uint8_t* __restrict__ output_image, int width, int height, uint8_t bitmask);


/* Derivatives */
__global__ void fod(uint8_t* __restrict__ heightmap, int* __restrict__ fods, float* __restrict__ fod_dirs, int width, int height);

__global__ void watershed(int* __restrict__ fods, bool* __restrict__ watersheds, int width, int height);


/*Local maxima */
__global__ void non_maximum_suppression(uint8_t* __restrict__ heightmap, float* __restrict__ fod_dirs, bool* __restrict__ watershed, bool* __restrict__ suppressed, int width, int height);

__global__ void local_max_8dir(uint8_t* __restrict__ heightmap, uint8_t* __restrict__ local_max_8dirs, int width, int height);

__global__ void local_max_4dir(const uint8_t* __restrict__ heightmap, uint8_t* __restrict__ local_max_4dirs, int width, int height);


/*Cone maps */
__global__ void create_cone_map_analytic(uint8_t* __restrict__ heightmap, bool* __restrict__ suppressed, float* __restrict__ fod_dirs, int* __restrict__ fods, uint8_t* __restrict__ cone_map, int width, int height);

__global__ void create_cone_map_analytic_local_mem(uint8_t* __restrict__ heightmap, bool* __restrict__ suppressed, float* __restrict__ fod_dirs, int* __restrict__ fods, uint8_t* __restrict__ cone_map, int width, int height);

__global__ void create_cone_map_8dir(uint8_t* __restrict__ heightmap, uint8_t* __restrict__ local_max_8dirs, uint8_t* __restrict__ cone_map, int width, int height);

__global__ void create_cone_map_8dir_local_mem(uint8_t* __restrict__ heightmap, uint8_t* __restrict__ local_max_8dirs, uint8_t* __restrict__ cone_map, int width, int height);

__global__ void create_cone_map_4dir_local_mem(uint8_t* __restrict__ heightmap, uint16_t* __restrict__ packed, uint8_t* __restrict__ cone_map, int width, int height);
