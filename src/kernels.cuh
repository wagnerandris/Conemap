#pragma once

#include <cstdint>

/* Structs */
struct AnalyticData {
	uint8_t h;
	bool watershed;
	float fod_dir;
};

struct DiscreteData8Dirs {
	uint8_t h;
	uint8_t dirs;
};

struct DiscreteData4Dirs {
	uint8_t h;
	uint8_t dirs;
};

struct IndexedAnalyticData {
	uint8_t x;
	uint8_t y;
	uint8_t h;
	bool watershed;
	float fod_dir;
};

struct IndexedDiscreteData8Dirs {
	uint8_t x;
	uint8_t y;
	uint8_t h;
	uint8_t dirs;
};

struct IndexedDiscreteData4Dirs {
	uint8_t x;
	uint8_t y;
	uint8_t h;
	uint8_t dirs;
};


/* Utils */

__global__ void invert(uint8_t* __restrict__ data, const int width, const int height);

__global__ void bits_to_image(const uint8_t* __restrict__ data, const uint8_t bitmask, uint8_t* __restrict__ output_image, const int width, const int height);


/* Derivatives / Local maxima */
__global__ void fod(const uint8_t* __restrict__ heightmap, int* __restrict__ fods, float* __restrict__ fod_dirs, const int width, const int height);

__global__ void watershed(const int* __restrict__ fods, bool* __restrict__ watersheds, const int width, const int height);

__global__ void non_maximum_suppression(const uint8_t* __restrict__ heightmap, const float* __restrict__ fod_dirs, const bool* __restrict__ watersheds, bool* __restrict__ suppressed, const int width, const int height);

__global__ void local_max_8dir(const uint8_t* __restrict__ heightmap, uint8_t* __restrict__ local_max_8dirs, const int width, const int height);

__global__ void local_max_4dir(const uint8_t* __restrict__ heightmap, uint8_t* __restrict__ local_max_4dirs, const int width, const int height);


/* Packing */

__global__ void pack(const uint8_t* __restrict__ heightmap, const bool* __restrict__ watersheds, const float* __restrict__ fod_dirs, AnalyticData* __restrict__ packed, const int width, const int height);

__global__ void pack(const uint8_t* __restrict__ heightmap, const bool* __restrict__ watersheds, const float* __restrict__ fod_dirs, IndexedAnalyticData* __restrict__ packed, const int width, const int height);

template<typename Data>
__global__ void pack_discrete(const uint8_t* __restrict__ heightmap, Data* __restrict__ packed, const int width, const int height);

template<typename Data>
__global__ void pack_discrete_continuously(const uint8_t* __restrict__ heightmap, Data* __restrict__ packed, const int width, const int height);


/* Cone maps */

//Baseline
__global__ void create_cone_map_baseline(const uint8_t* __restrict__ heightmap, uint8_t* __restrict__ cone_map, const int width, const int height);

//Preprocessed
__global__ void create_cone_map_analytic(const uint8_t* __restrict__ heightmap, const bool* __restrict__ watershed, const float* __restrict__ fod_dirs, const int* __restrict__ fods, uint8_t* __restrict__ cone_map, const int width, const int height);

__global__ void create_cone_map_8dir(const uint8_t* __restrict__ heightmap, const uint8_t* __restrict__ local_max_8dirs, uint8_t* __restrict__ cone_map, const int width, const int height);

__global__ void create_cone_map_4dir(const uint8_t* __restrict__ heightmap, const uint8_t* __restrict__ local_max_4dirs, uint8_t* __restrict__ cone_map, const int width, const int height);

//Shared memory
__global__ void create_cone_map_analytic_shared_mem(const uint8_t* __restrict__ heightmap, const bool* __restrict__ watershed, const float* __restrict__ fod_dirs, const int* __restrict__ fods, uint8_t* __restrict__ cone_map, const int width, const int height);

__global__ void create_cone_map_8dir_shared_mem(const uint8_t* __restrict__ heightmap, const uint8_t* __restrict__ local_max_8dirs, uint8_t* __restrict__ cone_map, const int width, const int height);

__global__ void create_cone_map_4dir_shared_mem(const uint8_t* __restrict__ heightmap, const uint8_t* __restrict__ local_max_4dirs, uint8_t* __restrict__ cone_map, const int width, const int height);

//Packed data
__global__ void create_cone_map_analytic_packed(const uint8_t* __restrict__ heightmap, const AnalyticData* __restrict__ packed, const int* __restrict__ fods, uint8_t* __restrict__ cone_map, const int width, const int height);

template<typename Data>
__global__ void create_cone_map_discrete_packed(const uint8_t* __restrict__ heightmap, const Data* __restrict__ packed, uint8_t* __restrict__ cone_map, const int width, const int height);

//Continuously packed data
__global__ void create_cone_map_analytic_continuous(const uint8_t* __restrict__ heightmap, const IndexedAnalyticData* __restrict__ packed, const int* __restrict__ fods, uint8_t* __restrict__ cone_map, const int width, const int height);

template<typename Data>
__global__ void create_cone_map_discrete_continuous(const uint8_t* __restrict__ heightmap, const Data* __restrict__ packed, uint8_t* __restrict__ cone_map, const int width, const int height);


#include "kernels.cu"
