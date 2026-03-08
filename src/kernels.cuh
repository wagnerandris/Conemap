#pragma once


/* Utils */
__global__ void pack(uint8_t* heightmap, uint8_t* local_max_4dirs, uint16_t* packed, int width, int height);

__global__ void invert(uint8_t* data, int width, int height);

__global__ void bits_to_image(uint8_t* data, uint8_t* output_image, int width, int height, uint8_t bitmask);


/* Derivatives */
__global__ void fod(uint8_t* heightmap, int* fods, float* fod_dirs, int width, int height);

__global__ void watershed(int* fods, bool* watersheds, int width, int height);


/* Local maxima */
__global__ void non_maximum_suppression(uint8_t* heightmap, float* fod_dirs, bool* watershed, bool* suppressed, int width, int height);

__global__ void local_max_8dir(uint8_t* heightmap, uint8_t* local_max_8dirs, int width, int height);


/* Cone maps */
__global__ void create_cone_map_analytic(uint8_t* heightmap, bool* suppressed, float* fod_dirs, int* fods, uint8_t* cone_map, int width, int height);

__global__ void create_cone_map_analytic_local_mem(uint8_t* heightmap, bool* suppressed, float* fod_dirs, int* fods, uint8_t* cone_map, int width, int height);

__global__ void create_cone_map_8dir_local_mem(uint8_t* heightmap, uint8_t* local_max_8dirs, uint8_t* cone_map, int width, int height);

__global__ void create_cone_map_8dir(uint8_t* heightmap, uint8_t* local_max_8dirs, uint8_t* cone_map, int width, int height);

__global__ void create_cone_map_4dir_local_mem(uint8_t* heightmap, uint16_t* packed, uint8_t* cone_map, int width, int height);
