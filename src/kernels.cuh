#pragma once


/* Utils */
__global__ void pack(const uint8_t* __restrict__ heightmap, uint16_t* __restrict__ packed, const int width, const int height);

__global__ void pack_continuously(const uint8_t* __restrict__ heightmap, uint16_t* __restrict__ packed, const int width, const int height);

__global__ void invert(uint8_t* __restrict__ data, const int width, const int height);

__global__ void bits_to_image(const uint8_t* __restrict__ data, uint8_t* __restrict__ output_image, const int width, const int height, uint8_t bitmask);


/* Derivatives */
__global__ void fod(const uint8_t* __restrict__ heightmap, int* __restrict__ fods, float* __restrict__ fod_dirs, const int width, const int height);

__global__ void watershed(const int* __restrict__ fods, bool* __restrict__ watersheds, const int width, const int height);


/*Local maxima */
__global__ void non_maximum_suppression(const uint8_t* __restrict__ heightmap, const float* __restrict__ fod_dirs, const bool* __restrict__ watershed, bool* __restrict__ suppressed, const int width, const int height);

__global__ void local_max_8dir(const uint8_t* __restrict__ heightmap, uint8_t* __restrict__ local_max_8dirs, const int width, const int height);

__global__ void local_max_4dir(const uint8_t* __restrict__ heightmap, uint8_t* __restrict__ local_max_4dirs, const int width, const int height);


/*Cone maps */
//Original
__global__ void create_cone_map_analytic(const uint8_t* __restrict__ heightmap, const bool* __restrict__ suppressed, const float* __restrict__ fod_dirs, const int* __restrict__ fods, uint8_t* __restrict__ cone_map, const int width, const int height);

__global__ void create_cone_map_8dir(const uint8_t* __restrict__ heightmap, const uint8_t* __restrict__ local_max_8dirs, uint8_t* __restrict__ cone_map, const int width, const int height);

//shared memory
__global__ void create_cone_map_analytic_shared_mem(const uint8_t* __restrict__ heightmap, const bool* __restrict__ suppressed, const float* __restrict__ fod_dirs, const int* __restrict__ fods, uint8_t* __restrict__ cone_map, const int width, const int height);

__global__ void create_cone_map_8dir_shared_mem(const uint8_t* __restrict__ heightmap, const uint8_t* __restrict__ local_max_8dirs, uint8_t* __restrict__ cone_map, const int width, const int height);

//continuously packed data
__global__ void create_cone_map_analytic_packed(const uint8_t* __restrict__ heightmap, const uint16_t* __restrict__ packed, uint8_t* __restrict__ cone_map, const int width, const int height);

__global__ void create_cone_map_8dir_packed(const uint8_t* __restrict__ heightmap, const uint16_t* __restrict__ packed, uint8_t* __restrict__ cone_map, const int width, const int height);

//compressed data
__global__ void create_cone_map_compressed(const uint8_t* __restrict__ heightmap, const uint16_t* __restrict__ packed, uint8_t* __restrict__ cone_map, const int width, const int height);
