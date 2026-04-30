#pragma once

// STD
#include <cmath>
#include <cstdint>

// CUDA
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "kernels.cuh"

/* Device utility functions */

__device__ __forceinline__ int index(const int width, const int height, const int u, const int v)
{
	return (v % height + height) % height * width + (u % width + width) % width;
}


/* Derivatives / Local maxima */

__device__ __constant__ int hkernel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

__device__ __constant__ int vkernel[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

__device__ __forceinline__ int2 sobel(const uint8_t *__restrict__ heightmap, const int u, const int v, const int width, const int height)
{
	int hsum = 0;
	int vsum = 0;
	for (int dv = 0; dv < 3; ++dv) {
		for (int du = 0; du < 3; ++du) {
			int didx = index(width, height, u + du - 1, v + dv - 1);
			hsum += heightmap[didx] * hkernel[dv * 3 + du];
			vsum += heightmap[didx] * vkernel[dv * 3 + du];
		}
	}
	return {hsum, vsum};
}

__device__ __forceinline__ uint8_t texel_local_max_8dir(const uint8_t* __restrict__ heightmap,
																												const uint8_t h,
																												const int u, const int v,
																												const int width, const int height)
{
	// Direction vectors: Right, Up-Right, Up, Up-Left, Left, Down-Left, Down, Down-Right
	const int du[8] = {1, 1, 0, -1, -1, -1, 0, 1};
	const int dv[8] = {0, -1, -1, -1, 0, 1, 1, 1};

	uint8_t result = 0;
	for (int dir = 0; dir < 8; ++dir) {
		// local max given direction (if there is a plateau, we need its last point)
		if (h > heightmap[index(width, height, u + du[dir], v + dv[dir])] &&
				h >= heightmap[index(width, height, u - du[dir], v - dv[dir])]) {
			result |= (1 << dir);
		}
	}
	return result;
}

__device__ __forceinline__ uint8_t texel_local_max_4dir(const uint8_t* __restrict__ heightmap,
																												const uint8_t h,
																												const int u, const int v,
																												const int width, const int height)
{
	// Direction vectors: Right, Up, Left, Down
	const int du[4] = {1, 0, -1, 0};
	const int dv[4] = {0, -1, 0, 1};

	uint8_t result = 0;
	for (int dir = 0; dir < 4; ++dir) {
		// local max given direction (if there is a plateau, we need its last point)
		if (h > heightmap[index(width, height, u + du[dir], v + dv[dir])] &&
				h >= heightmap[index(width, height, u - du[dir], v - dv[dir])]) {
			result |= (1 << dir);
		}
	}
	return result;
}


/* Packing */

__device__ __forceinline__ void pack_texel(const uint8_t* __restrict__ heightmap,
																					 const int u, const int v,
																					 const int width, const int height,
																					 DiscreteData8Dirs& data)
{
	uint8_t h = heightmap[index(width, height, u, v)];
	data = {h,
					texel_local_max_8dir(heightmap, h, u, v, width, height)};
}

__device__ __forceinline__ void pack_texel(const uint8_t* __restrict__ heightmap,
																					 const int u, const int v,
																					 const int width, const int height,
																					 DiscreteData4Dirs& data)
{
	uint8_t h = heightmap[index(width, height, u, v)];
	data = {h,
					texel_local_max_4dir(heightmap, h, u, v, width, height)};
}

__device__ __forceinline__ bool pack_texel(const uint8_t* __restrict__ heightmap,
																					 const int u, const int v,
																					 const int width, const int height,
																					 IndexedDiscreteData8Dirs& data)
{
	uint8_t h = heightmap[index(width, height, u, v)];
	uint8_t dirs = texel_local_max_8dir(heightmap, h, u, v, width, height);
	data = {static_cast<uint8_t>(threadIdx.x),
					static_cast<uint8_t>(threadIdx.y),
					h,
					dirs};
	return dirs != 0; // limiting point in any direction
}

__device__ __forceinline__ bool pack_texel(const uint8_t* __restrict__ heightmap,
																					 const int u, const int v,
																					 const int width, const int height,
																					 IndexedDiscreteData4Dirs& data)
{
	uint8_t h = heightmap[index(width, height, u, v)];
	uint8_t dirs = texel_local_max_4dir(heightmap, h, u, v, width, height);
	data = {static_cast<uint8_t>(threadIdx.x),
					static_cast<uint8_t>(threadIdx.y),
					h,
					dirs};
	return dirs != 0; // limiting point in any direction
}

__device__ __forceinline__ bool pack_texel(const uint8_t* __restrict__ heightmap,
																					 const int u, const int v,
																					 const int width, const int height,
																					 uint16_t& data)
{
	uint8_t h = heightmap[index(width, height, u, v)];
	uint8_t dirs = texel_local_max_4dir(heightmap, h, u, v, width, height);
	data = ((threadIdx.x & 0x7) << 13) |
				 ((threadIdx.y & 0x7) << 10) |
				 ((h           & 0xFC) << 2) |
				 (dirs         & 0xF);
	return dirs != 0; // limiting point in any direction
}


/* Cone map generation */

__device__ __forceinline__ int dir8(const int du, const int dv)
{
	const float t = 0.41421356237f;

	float adu = static_cast<float>(abs(du));
	float adv = static_cast<float>(abs(dv));

	return (adv <= adu * t)		? (du > 0 ? 0 : 4)
				 : (adu <= adv * t) ? (dv > 0 ? 2 : 6)
				 : (du > 0)					? (dv > 0 ? 1 : 7)
														: (dv > 0 ? 3 : 5);
}

__device__ __forceinline__ int dir4(const int du, const int dv)
{
	if (abs(du) > abs(dv))
		return du > 0 ? 0 : 2; // E / W
	else
		return dv > 0 ? 1 : 3; // N / S
}

__device__ __forceinline__ int2 get_search_block_idx(const int r, const int step)
{
	int2 bidx;
	if (step < 2 * r) {
		// Right side:
		bidx.x = blockIdx.x + r;
		bidx.y = blockIdx.y - r + step;
	} else if (step < 4 * r) {
		// Top side:
		bidx.x = blockIdx.x - r + (step - 2 * r);
		bidx.y = blockIdx.y - r;
	} else if (step < 6 * r) {
		// Left side:
		bidx.x = blockIdx.x - r;
		bidx.y = blockIdx.y + r - (step - 4 * r);
	} else {
		// Bottom side:
		bidx.x = blockIdx.x + r - (step - 6 * r);
		bidx.y = blockIdx.y + r;
	}
	return bidx;
}

__device__ __forceinline__ bool not_empty(uint16_t data)
{
	return data != 0;
}

__device__ __forceinline__ bool not_empty(IndexedAnalyticData data)
{
	return data.watershed;
}

template<typename Data>
__device__ __forceinline__ bool not_empty(Data data)
{
	return data.dirs != 0;
}

__device__ __forceinline__ void limit_cone_baseline(const uint8_t *heightmap,
																										const int width, const int height,
																										const int u, const int v,
																										const int cu, const int cv,
																										const float iwidth, const float iheight,
																										const float h, float &ratio2)
{
	int du = cu - u > 0 ? 1 : -1;
	int dv = cv - v > 0 ? 1 : -1;
	
	// check if limiting point, skip if not
	uint8_t h00 = heightmap[index(width, height, cu, cv)];
	uint8_t h01 = heightmap[index(width, height, cu, cv + dv)];
	uint8_t h10 = heightmap[index(width, height, cu + du, cv)];
	uint8_t h11 = heightmap[index(width, height, cu + du, cv + dv)];

	if (h00 <= h10 && h00 <= h01 && h10 <= h11 && h01 <= h11) return;

	// normalize u and v displacements
	const float dun = (cu - u) * iwidth;
	const float dvn = (cv - v) * iheight;

	const float d2 = dun * dun + dvn * dvn; // distance squared
	const float dh = h00 / 255.0f - h; // height difference
	const float dh2 = dh * dh;

	if (dh > 0.0f && dh2 * ratio2 > d2) // if steeper than current
		ratio2 = d2 / dh2; // override squared ratio
}

__device__ __forceinline__ void limit_cone(const uint8_t *heightmap,
																					 const int width, const int height,
																					 const int u, const int v,
																					 const int cu, const int cv,
																					 const float iwidth, const float iheight,
																					 const float h, float &ratio2)
{
	// normalize u and v displacements
	const float dun = (cu - u) * iwidth;
	const float dvn = (cv - v) * iheight;

	const float d2 = dun * dun + dvn * dvn; // distance squared
	const float dh = heightmap[index(width, height, cu, cv)] / 255.0f - h; // height difference
	const float dh2 = dh * dh;

	if (dh > 0.0f && dh2 * ratio2 > d2) // if steeper than current
		ratio2 = d2 / dh2; // override squared ratio
}

__device__ __forceinline__ void limit_cone(const uint8_t ch,
																					 const bool watershed,
																					 const float fod_dir,
																					 const int u, const int v,
																					 const float iwidth, const float iheight,
																					 const int bx, const int by,
																					 const int k,
																					 const float h,
																					 float& ratio2)
{
	const int du = bx * 8 + (k % 8) - u;
	const int dv = by * 8 + (k / 8) - v;

	if (!watershed ||
			abs(remainderf((fod_dir - atan2f(dv, du) - M_PI_2), M_PI)) > (M_PI / 8.0f))
		return;

	// normalize u and v displacements
	const float dun = du * iwidth;
	const float dvn = dv * iheight;

	const float d2 = dun * dun + dvn * dvn; // distance squared
	const float dh = ch / 255.0 - h;
	const float dh2 = dh * dh;

	if (dh > 0.0f && dh2 * ratio2 > d2) // if steeper than current
		ratio2 = d2 / dh2; // override squared ratio
}

__device__ __forceinline__ void limit_cone_8dirs(const uint8_t ch,
																					 		 	 const uint8_t local_max_8dir,
																					 		 	 const int u, const int v,
																					 		 	 const float iwidth, const float iheight,
																					 		 	 const int bx, const int by,
																					 		 	 const int k,
																					 		 	 const float h,
																					 		 	 float& ratio2)
{
	const int du = bx * 8 + (k % 8) - u;
	const int dv = by * 8 + (k / 8) - v;
	const int discrete_dir = dir8(du, dv); // direction of texel from cone apex

	if (!(local_max_8dir & (1 << discrete_dir)))
		return; // not limiting point in this direction

	// normalize u and v displacements
	const float dun = du * iwidth;
	const float dvn = dv * iheight;

	const float d2 = dun * dun + dvn * dvn; // distance squared
	const float dh = ch / 255.0f - h; // height difference
	const float dh2 = dh * dh;

	if (dh > 0.0f && dh2 * ratio2 > d2) // if steeper than current
		ratio2 = d2 / dh2; // override squared ratio
}

__device__ __forceinline__ void limit_cone_4dirs(const uint8_t ch,
																					 		 	 const uint8_t local_max_4dir,
																					 		 	 const int u, const int v,
																					 		 	 const float iwidth, const float iheight,
																					 		 	 const int bx, const int by,
																					 		 	 const int k,
																					 		 	 const float h,
																					 		 	 float& ratio2)
{
	const int du = bx * 8 + (k % 8) - u;
	const int dv = by * 8 + (k / 8) - v;
	const int discrete_dir = dir4(du, dv); // direction of texel from cone apex

	if (!(local_max_4dir & (1 << discrete_dir)))
		return; // not limiting point in this direction

	// normalize u and v displacements
	const float dun = du * iwidth;
	const float dvn = dv * iheight;

	const float d2 = dun * dun + dvn * dvn; // distance squared
	const float dh = ch / 255.0f - h; // height difference
	const float dh2 = dh * dh;

	if (dh > 0.0f && dh2 * ratio2 > d2) // if steeper than current
		ratio2 = d2 / dh2; // override squared ratio
}


__device__ __forceinline__ void limit_cone(const AnalyticData packed,
																					 const int u, const int v,
																					 const float iwidth, const float iheight,
																					 const int bx, const int by,
																					 const int k,
																					 const float h,
																					 float& ratio2)
{
	const int du = bx * 8 + (k % 8) - u;
	const int dv = by * 8 + (k / 8) - v;
	const int discrete_dir = dir8(du, dv); // direction of texel from cone apex

	if (!packed.watershed ||
			abs(remainderf((packed.fod_dir - atan2f(dv, du) - M_PI_2), M_PI)) > (M_PI / 8.0f))
		return;

	// normalize u and v displacements
	const float dun = du * iwidth;
	const float dvn = dv * iheight;

	const float d2 = dun * dun + dvn * dvn; // distance squared
	const float dh = packed.h / 255.0f - h; // height difference
	const float dh2 = dh * dh;

	if (dh > 0.0f && dh2 * ratio2 > d2) // if steeper than current
		ratio2 = d2 / dh2; // override squared ratio
}

__device__ __forceinline__ void limit_cone(const DiscreteData8Dirs packed,
																					 const int u, const int v,
																					 const float iwidth, const float iheight,
																					 const int bx, const int by,
																					 const int k,
																					 const float h,
																					 float& ratio2)
{
	const int du = bx * 8 + (k % 8) - u;
	const int dv = by * 8 + (k / 8) - v;
	const int discrete_dir = dir8(du, dv); // direction of texel from cone apex

	if (!(packed.dirs & (1 << discrete_dir)))
		return; // not limiting point in this direction

	// normalize u and v displacements
	const float dun = du * iwidth;
	const float dvn = dv * iheight;

	const float d2 = dun * dun + dvn * dvn; // distance squared
	const float dh = packed.h / 255.0f - h; // height difference
	const float dh2 = dh * dh;

	if (dh > 0.0f && dh2 * ratio2 > d2) // if steeper than current
		ratio2 = d2 / dh2; // override squared ratio
}

__device__ __forceinline__ void limit_cone(const DiscreteData4Dirs packed,
																					 const int u, const int v,
																					 const float iwidth, const float iheight,
																					 const int bx, const int by,
																					 const int k,
																					 const float h,
																					 float& ratio2)
{
	const int du = bx * 8 + (k % 8) - u;
	const int dv = by * 8 + (k / 8) - v;
	const int discrete_dir = dir4(du, dv); // direction of texel from cone apex

	if (!(packed.dirs & (1 << discrete_dir)))
		return; // not limiting point in this direction

	// normalize u and v displacements
	const float dun = du * iwidth;
	const float dvn = dv * iheight;

	const float d2 = dun * dun + dvn * dvn; // distance squared
	const float dh = packed.h / 255.0f - h; // height difference
	const float dh2 = dh * dh;

	if (dh > 0.0f && dh2 * ratio2 > d2) // if steeper than current
		ratio2 = d2 / dh2; // override squared ratio
}

__device__ __forceinline__ void limit_cone(const uint16_t compressed,
																					 const int u, const int v,
																					 const float iwidth, const float iheight,
																					 const int bx, const int by,
																					 const int k,
																					 const float h,
																					 float& ratio2)
{
	const int du = bx * 8 + (k % 8) - u;
	const int dv = by * 8 + (k / 8) - v;
	const int discrete_dir = dir4(du, dv); // direction of texel from cone apex

	if (!(compressed & (1 << discrete_dir)))
		return; // not limiting point in this direction

	// normalize u and v displacements
	const float dun = du * iwidth;
	const float dvn = dv * iheight;

	const float d2 = dun * dun + dvn * dvn; // distance squared
	const float dh = ((compressed >> 2) & 0xFC) / 255.0f - h; // height difference
	const float dh2 = dh * dh;

	if (dh > 0.0f && dh2 * ratio2 > d2) // if steeper than current
		ratio2 = d2 / dh2; // override squared ratio
}

__device__ __forceinline__ void limit_cone(const IndexedAnalyticData packed,
																					 const int u, const int v,
																					 const float iwidth, const float iheight,
																					 const int bx, const int by,
																					 const float h, float& ratio2)
{
	const int du = bx * 8 + packed.x - u;
	const int dv = by * 8 + packed.y - v;
	const int discrete_dir = dir8(du, dv); // direction of texel from cone apex

	if (!packed.watershed ||
			abs(remainderf((packed.fod_dir - atan2f(dv, du) - M_PI_2), M_PI)) > (M_PI / 8.0f))
		return;

	// normalize u and v displacements
	const float dun = du * iwidth;
	const float dvn = dv * iheight;

	const float d2 = dun * dun + dvn * dvn; // distance squared
	const float dh = packed.h / 255.0f - h; // height difference
	const float dh2 = dh * dh;

	if (dh > 0.0f && dh2 * ratio2 > d2) // if steeper than current
		ratio2 = d2 / dh2; // override squared ratio
}

__device__ __forceinline__ void limit_cone(const IndexedDiscreteData8Dirs packed,
																					 const int u, const int v,
																					 const float iwidth, const float iheight,
																					 const int bx, const int by,
																					 const float h, float& ratio2)
{
	const int du = bx * 8 + packed.x - u;
	const int dv = by * 8 + packed.y - v;
	const int discrete_dir = dir8(du, dv); // direction of texel from cone apex

	if (!(packed.dirs & (1 << discrete_dir)))
		return; // not limiting point in this direction

	// normalize u and v displacements
	const float dun = du * iwidth;
	const float dvn = dv * iheight;

	const float d2 = dun * dun + dvn * dvn; // distance squared
	const float dh = packed.h / 255.0f - h; // height difference
	const float dh2 = dh * dh;

	if (dh > 0.0f && dh2 * ratio2 > d2) // if steeper than current
		ratio2 = d2 / dh2; // override squared ratio
}

__device__ __forceinline__ void limit_cone(const IndexedDiscreteData4Dirs packed,
																					 const int u, const int v,
																					 const float iwidth, const float iheight,
																					 const int bx, const int by,
																					 const float h, float& ratio2)
{
	const int du = bx * 8 + packed.x - u;
	const int dv = by * 8 + packed.y - v;
	const int discrete_dir = dir4(du, dv); // direction of texel from cone apex

	if (!(packed.dirs & (1 << discrete_dir)))
		return; // not limiting point in this direction

	// normalize u and v displacements
	const float dun = du * iwidth;
	const float dvn = dv * iheight;

	const float d2 = dun * dun + dvn * dvn; // distance squared
	const float dh = packed.h / 255.0f - h; // height difference
	const float dh2 = dh * dh;

	if (dh > 0.0f && dh2 * ratio2 > d2) // if steeper than current
		ratio2 = d2 / dh2; // override squared ratio
}

__device__ __forceinline__ void limit_cone(const uint16_t compressed,
																					 const int u, const int v,
																					 const float iwidth, const float iheight,
																					 const int bx, const int by,
																					 const float h, float& ratio2)
{
	const int du = bx * 8 + (compressed >> 13) - u;
	const int dv = by * 8 + ((compressed >> 10) & 7) - v;
	const int discrete_dir = dir4(du, dv); // direction of texel from cone apex

	if (!(compressed & (1 << discrete_dir)))
		return; // not limiting point in this direction

	// normalize u and v displacements
	const float dun = du * iwidth;
	const float dvn = dv * iheight;

	const float d2 = dun * dun + dvn * dvn; // distance squared
	const float dh = ((compressed >> 2) & 0xFC) / 255.0f - h; // height difference
	const float dh2 = dh * dh;

	if (dh > 0.0f && dh2 * ratio2 > d2) // if steeper than current
		ratio2 = d2 / dh2; // override squared ratio
}


/* ------------------- Global kernels ------------------ */


/* Utils */

__global__ void invert(uint8_t* __restrict__ data, const int width, const int height)
{
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height)
		return;

	data[v * width + u] = 255 - data[v * width + u];
}

__global__ void bits_to_image(const uint8_t* __restrict__ data,
															const uint8_t bitmask,
															uint8_t* __restrict__ output_image,
															const int width, const int height)
{
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height)
		return;

	output_image[v * width + u] = data[v * width + u] & bitmask ? 255 : 0;
}

/* Derivatives / Local maxima */
__global__ void fod(const uint8_t* __restrict__ heightmap,
										int* __restrict__ fods,
										float* __restrict__ fod_dirs,
										const int width, const int height)
{
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height)
		return;

	int idx = v * width + u;

	int2 sums = sobel(heightmap, u, v, width, height);

	fod_dirs[idx] = atan2f(sums.y, sums.x);

	fods[idx * 2 + 0] = sums.x;
	fods[idx * 2 + 1] = sums.y;
}

__global__ void watershed(const int* __restrict__ fods,
													bool* __restrict__ watersheds,
													const int width, const int height)
{
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height)
		return;

	// Second order derivative
	int hhsum = 0;
	int vhsum = 0;
	int hvsum = 0;
	int vvsum = 0;
	for (int dv = 0; dv < 3; ++dv) {
		for (int du = 0; du < 3; ++du) {
			int didx = index(width, height, u + du - 1, v + dv - 1) * 2;
			hhsum += fods[didx + 0] * hkernel[dv * 3 + du];
			hvsum += fods[didx + 1] * hkernel[dv * 3 + du];
			vhsum += fods[didx + 0] * vkernel[dv * 3 + du];
			vvsum += fods[didx + 1] * vkernel[dv * 3 + du];
		}
	}

	int idx = v * width + u;

	watersheds[idx] =
			hhsum * fods[idx * 2 + 1] * fods[idx * 2 + 1] -
					(hvsum + vhsum) * fods[idx * 2 + 0] * fods[idx * 2 + 1] +
					vvsum * fods[idx * 2 + 0] * fods[idx * 2 + 0] <
			0;
}

__global__ void non_maximum_suppression(const uint8_t* __restrict__ heightmap,
																				const float* __restrict__ fod_dirs,
																				const bool* __restrict__ watersheds,
																				bool* __restrict__ suppressed,
																				const int width, const int height)
{
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height)
		return;

	int idx = v * width + u;

	if (!watersheds[idx]) {
		suppressed[idx] = 0;
		return;
	}

	// get the nearest discrete direction
	uint8_t ddir = static_cast<uint8_t>((fod_dirs[idx] + M_PI // all positive
																			 + M_PI_4f / 2.0f)		// align regions
																			/ M_PI_4f						 // 8 dirs
																			) %
								 4; // opposite ones are the same

	// neighbours orthogonal to ddir
	const int du[4] = {0, -1, 1, 1};
	const int dv[4] = {1, 1, 0, 1};

	// if one of the neighbours has greater height, suppress the current texel
	if (heightmap[index(width, height, u + du[ddir], v + dv[ddir])] >
			heightmap[idx])
		suppressed[idx] = 0;
	else if (heightmap[index(width, height, u - du[ddir], v - dv[ddir])] >
					 heightmap[idx])
		suppressed[idx] = 0;
	else
		suppressed[idx] = watersheds[idx];
}

__global__ void local_max_8dir(const uint8_t* __restrict__ heightmap,
															 uint8_t* __restrict__ local_max_8dirs,
															 const int width, const int height)
{
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height)
		return;

	int idx = v * width + u;
	const int h = heightmap[idx];
	local_max_8dirs[idx] = texel_local_max_8dir(heightmap, h, u, v, width, height);
}

__global__ void local_max_4dir(const uint8_t* __restrict__ heightmap,
															 uint8_t* __restrict__ local_max_4dirs,
															 const int width, const int height)
{
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height)
		return;

	int idx = v * width + u;
	int h = heightmap[idx];
	local_max_4dirs[idx] = texel_local_max_4dir(heightmap, h, u, v, width, height);
}

/* Packing */

__global__ void pack(const uint8_t* __restrict__ heightmap,
										 const bool* __restrict__ watersheds,
										 const float* __restrict__ fod_dirs,
										 AnalyticData* __restrict__ packed,
										 const int width, const int height)
{
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = v * width + u;

	if (u >= width || v >= height)
		return;

	packed[idx] = {heightmap[idx],
								 watersheds[idx],
								 fod_dirs[idx]};
}

__global__ void pack(const uint8_t* __restrict__ heightmap,
										 const bool* __restrict__ watersheds,
										 const float* __restrict__ fod_dirs,
										 IndexedAnalyticData* __restrict__ packed,
										 const int width, const int height)
{
	__shared__ uint8_t warp0_count;

	// global indices
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = v * width + u;
	const int block_base = (blockIdx.y * blockDim.y) * width + (blockIdx.x * blockDim.x);

	// local indices
	const int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
	const int lane_idx = local_idx & 31;
	const int warp_idx = local_idx >> 5;

	bool limiting = false;
	IndexedAnalyticData data;

	if (u < width && v < height) {
		data = {static_cast<uint8_t>(threadIdx.x),
						static_cast<uint8_t>(threadIdx.y),
						heightmap[idx],
						watersheds[idx],
						fod_dirs[idx]};
		limiting = watersheds[idx];
	}

	// Find place in continuous memory and write there
	unsigned ballot = __ballot_sync(0xFFFFFFFF, limiting);
	int local_rank = __popc(ballot & ((1u << lane_idx) - 1));
	int warp_count = __popc(ballot);

	if (local_idx == 0)
		warp0_count = warp_count;
	__syncthreads();

	int block_rank = (warp_idx == 0 ? 0 : warp0_count) + local_rank;

	if (limiting) {
		int out_idx = block_base + (block_rank / blockDim.x) * width + (block_rank % blockDim.x);
		packed[out_idx] = data;
	}

	// Set terminating null value
	if (local_idx == 32) { // first thread of second warp
		int block_count = warp0_count + warp_count;
		if (block_count < 64) {
			int out_idx = block_base + (block_count / blockDim.x) * width + (block_count % blockDim.x);
			packed[out_idx] = IndexedAnalyticData{};
		}
	}
}

template<typename Data>
__global__ void pack_discrete(const uint8_t* __restrict__ heightmap,
															Data* __restrict__ packed,
															const int width, const int height)
{
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height)
		return;

	pack_texel(heightmap, u, v, width, height, packed[v * width + u]);
}

template<typename Data>
__global__ void pack_discrete_continuously(const uint8_t* __restrict__ heightmap,
																					 Data* __restrict__ packed,
																					 const int width, const int height)
{
	__shared__ uint8_t warp0_count;

	// global indices
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	const int block_base = (blockIdx.y * blockDim.y) * width + (blockIdx.x * blockDim.x);

	// local indices
	const int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
	const int lane_idx = local_idx & 31;
	const int warp_idx = local_idx >> 5;

	bool limiting = false;
	Data data;

	if (u < width && v < height) { // if in bounds check
		limiting = pack_texel(heightmap, u, v, width, height, data);
	}

	// Find place in continuous memory and write there
	unsigned ballot = __ballot_sync(0xFFFFFFFF, limiting);
	int local_rank = __popc(ballot & ((1u << lane_idx) - 1));
	int warp_count = __popc(ballot);

	if (local_idx == 0)
		warp0_count = warp_count;
	__syncthreads();

	int block_rank = (warp_idx == 0 ? 0 : warp0_count) + local_rank;

	if (limiting) {
		int out_idx = block_base + (block_rank / blockDim.x) * width + (block_rank % blockDim.x);
		packed[out_idx] = data;
	}

	// Set terminating null value
	if (local_idx == 32) { // first thread of second warp
		int block_count = warp0_count + warp_count;
		if (block_count < 64) {
			int out_idx = block_base + (block_count / blockDim.x) * width + (block_count % blockDim.x);
			packed[out_idx] = Data{};
		}
	}
}

/*Cone maps */
__global__ void create_cone_map_baseline(const uint8_t* __restrict__ heightmap,
																				 uint8_t* __restrict__ cone_map,
																				 const int width, const int height)
{
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = v * width + u;

	if (u >= width || v >= height)
		return;

	// variables for cone setting
	const float iwidth = 1.0f / width;
	const float iheight = 1.0f / height;
	const float side = max(width, height);
	const float h = heightmap[idx] / 255.0f; // normalized height
	float ratio2 = 1.0f; // squared tangent of half aperture angle

	// init variables
	int cu, cv;

	// search with increasing radius around the texel
	for (int r = 1;
			 // otherwise we can't find anything steeper than the current min_ratio
			 // (see min_ratio assignment)
			 r * r <= (1.0f - h) * (1.0f - h) * ratio2 * max(width, height) * max(width, height);
			 ++r) {

		// Right side

		cu = u + r; // u displacement
		for (int cv = v - r; cv < v + r - 1; ++cv) { // go through side
			limit_cone_baseline(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h, ratio2);
		}

		// Top side

		cv = v - r; // u displacement
		for (int cu = u - r; cu < u + r - 1; ++cu) { // go through side
			limit_cone_baseline(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h, ratio2);
		}

		// Left side

		cu = u - r; // u displacement
		for (int cv = v - r + 1; cv < v + r; ++cv) { // go through side
			limit_cone_baseline(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h, ratio2);
		}

		// Bottom side

		cv = v + r; // u displacement
		for (int cu = u - r + 1; cu < u + r; ++cu) { // go through side
			limit_cone_baseline(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h, ratio2);
		}
	}

	float sqrt_ratio = sqrt(sqrt(ratio2));
	// most of the data is on the low end...sqrting again spreads it better
	// (plus multiply is a cheap operation in shaders!)
	// -- Dummer
	
	// First order derivatives
	const int2 sums = sobel(heightmap, u, v, width, height);
	const uint8_t dhdu = (sums.x + 1020) / 8;
	const uint8_t dhdv = (sums.y + 1020) / 8;

	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<uint8_t>(sqrt_ratio * 255);
	cone_map[idx * 4 + 2] = dhdu;
	cone_map[idx * 4 + 3] = dhdv;
}

//Preprocessed
__global__ void create_cone_map_analytic(const uint8_t* __restrict__ heightmap,
																				 const bool* __restrict__ watershed,
																				 const float* __restrict__ fod_dirs,
																				 const int* __restrict__ fods,
																				 uint8_t* __restrict__ cone_map,
																				 const int width, const int height)
{
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = v * width + u;

	if (u >= width || v >= height)
		return;

	// variables for cone setting
	const float iwidth = 1.0f / width;
	const float iheight = 1.0f / height;
	const float side = max(width, height);
	const float h = heightmap[idx] / 255.0f; // normalized height
	float ratio2 = 1.0f; // squared tangent of half aperture angle

	// init variables
	int cu, cv;

	// search with increasing radius around the texel
	for (int r = 1;
			 // otherwise we can't find anything steeper than the current min_ratio
			 // (see min_ratio assignment)
			 r * r <= (1.0f - h) * (1.0f - h) * ratio2 * max(width, height) * max(width, height);
			 ++r) {

		// Right side

		cu = u + r; // u displacement
		for (int cv = v - r; cv < v + r - 1; ++cv) { // go through side
			// check if (suppressed) watershed point, skip if not
			if (!watershed[index(width, height, cu, cv)] ||
					abs(remainderf((fod_dirs[index(width, height, cu, cv)] -
													atan2f(cv, cu) - M_PI_2),
												 M_PI)) > (M_PI / 8.0))
				continue;
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}

		// Top side

		cv = v - r; // u displacement
		for (int cu = u - r; cu < u + r - 1; ++cu) { // go through side
			// check if (suppressed) watershed point, skip if not
			if (!watershed[index(width, height, cu, cv)] ||
					abs(remainderf((fod_dirs[index(width, height, cu, cv)] -
													atan2f(cv, cu) - M_PI_2),
												 M_PI)) > (M_PI / 8.0))
				continue;
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}

		// Left side

		cu = u - r; // u displacement
		for (int cv = v - r + 1; cv < v + r; ++cv) { // go through side
			// check if (suppressed) watershed point, skip if not
			if (!watershed[index(width, height, cu, cv)] ||
					abs(remainderf((fod_dirs[index(width, height, cu, cv)] -
													atan2f(cv, cu) - M_PI_2),
												 M_PI)) > (M_PI / 8.0))
				continue;
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}

		// Bottom side

		cv = v + r; // u displacement
		for (int cu = u - r + 1; cu < u + r; ++cu) { // go through side
			// check if (suppressed) watershed point, skip if not
			if (!watershed[index(width, height, cu, cv)] ||
					abs(remainderf((fod_dirs[index(width, height, cu, cv)] -
													atan2f(cv, cu) - M_PI_2),
												 M_PI)) > (M_PI / 8.0))
				continue;
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}
	}

	float sqrt_ratio = sqrt(sqrt(ratio2));
	// most of the data is on the low end...sqrting again spreads it better
	// (plus multiply is a cheap operation in shaders!)
	// -- Dummer

	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<uint8_t>(sqrt_ratio * 255);
	cone_map[idx * 4 + 2] = (fods[idx * 2] + 1020) / 8;
	cone_map[idx * 4 + 3] = (fods[idx * 2 + 1] + 1020) / 8;
}

__global__ void create_cone_map_8dir(const uint8_t* __restrict__ heightmap,
																		 const uint8_t* __restrict__ local_max_8dirs,
																		 uint8_t* __restrict__ cone_map,
																		 const int width, const int height)
{
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = v * width + u;

	if (u >= width || v >= height)
		return;

	// variables for cone setting
	const float iwidth = 1.0f / width;
	const float iheight = 1.0f / height;
	const float side = max(width, height);
	const float h = heightmap[idx] / 255.0f; // normalized height
	float ratio2 = 1.0f; // squared tangent of half aperture angle

	// init variables
	int cu, cv;
	int start, end;

	// search with increasing radius around the texel
	for (int r = 1;
			 // otherwise we can't find anything steeper than the current min_ratio
			 // (see min_ratio assignment)
			 r * r <= (1.0f - h) * (1.0f - h) * ratio2 * max(width, height) *
										max(width, height);
			 ++r) {

		// Right side

		cu = u + r; // u displacement
		// set v limits
		start = v - r;
		end = v + r - 1;
		// go through side
		// check if local maxima in the given direction
		// skip if not
		for (int cv = start; cv <= start + r / 2; ++cv) {
			int discrete_dir = 1;
			if (!(local_max_8dirs[cv * width + cu] & 1 << discrete_dir))
				continue;
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}
		for (int cv = start + r / 2 + 1; cv < end - r / 2; ++cv) {
			int discrete_dir = 0;
			if (!(local_max_8dirs[cv * width + cu] & 1 << discrete_dir))
				continue;
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}
		for (int cv = end - r / 2; cv < end; ++cv) {
			int discrete_dir = 7;
			if (!(local_max_8dirs[cv * width + cu] & 1 << discrete_dir))
				continue;
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}

		// Top side

		cv = v - r; // v displacement
		// set u limits
		start = u - r;
		end = u + r - 1;
		// go through side
		// check if local maxima in the given direction
		// skip if not
		for (int cu = start; cu <= start + r / 2; ++cu) {
			int discrete_dir = 3;
			if (!(local_max_8dirs[cv * width + cu] & 1 << discrete_dir))
				continue;
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}
		for (int cu = start + r / 2 + 1; cu < end - r / 2; ++cu) {
			int discrete_dir = 2;
			if (!(local_max_8dirs[cv * width + cu] & 1 << discrete_dir))
				continue;
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}
		for (int cu = end - r / 2; cu < end; ++cu) {
			int discrete_dir = 1;
			if (!(local_max_8dirs[cv * width + cu] & 1 << discrete_dir))
				continue;
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}

		// Left side

		cu = u - r; // u displacement
		// set v limits
		start = v - r + 1;
		end = v + r;
		// go through side
		// check if local maxima in the given direction
		// skip if not
		for (int cv = start; cv <= start + r / 2; ++cv) {
			int discrete_dir = 3;
			if (!(local_max_8dirs[cv * width + cu] & 1 << discrete_dir))
				continue;
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}
		for (int cv = start + r / 2 + 1; cv < end - r / 2; ++cv) {
			int discrete_dir = 4;
			if (!(local_max_8dirs[cv * width + cu] & 1 << discrete_dir))
				continue;
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}
		for (int cv = end - r / 2; cv < end; ++cv) {
			int discrete_dir = 5;
			if (!(local_max_8dirs[cv * width + cu] & 1 << discrete_dir))
				continue;
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}

		// Bottom side

		cv = v + r; // v displacement
		// set u limits
		start = u - r + 1;
		end = u + r;
		// go through side
		// check if local maxima in the given direction
		// skip if not
		for (int cu = start; cu <= start + r / 2; ++cu) {
			int discrete_dir = 5;
			if (!(local_max_8dirs[cv * width + cu] & 1 << discrete_dir))
				continue;
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}
		for (int cu = start + r / 2 + 1; cu < end - r / 2; ++cu) {
			int discrete_dir = 6;
			if (!(local_max_8dirs[cv * width + cu] & 1 << discrete_dir))
				continue;
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}
		for (int cu = end - r / 2; cu < end; ++cu) {
			int discrete_dir = 7;
			if (!(local_max_8dirs[cv * width + cu] & 1 << discrete_dir))
				continue;
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}
	}

	// Square root of cone ratio -> must be sqared in rendering shader
	float sqrt_ratio = sqrt(sqrt(ratio2));


	// First order derivatives
	const int2 sums = sobel(heightmap, u, v, width, height);
	const uint8_t dhdu = (sums.x + 1020) / 8;
	const uint8_t dhdv = (sums.y + 1020) / 8;

	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<uint8_t>(sqrt_ratio * 255);
	cone_map[idx * 4 + 2] = dhdu;
	cone_map[idx * 4 + 3] = dhdv;
}

__global__ void create_cone_map_4dir(const uint8_t* __restrict__ heightmap,
																		 const uint8_t* __restrict__ local_max_4dirs,
																		 uint8_t* __restrict__ cone_map,
																		 const int width, const int height)
{
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = v * width + u;

	if (u >= width || v >= height)
		return;

	// variables for cone setting
	const float iwidth = 1.0f / width;
	const float iheight = 1.0f / height;
	const float side = max(width, height);
	const float h = heightmap[idx] / 255.0f; // normalized height
	float ratio2 = 1.0f; // squared tangent of half aperture angle

	// init variables
	int cu, cv;
	int start, end;

	// search with increasing radius around the texel
	for (int r = 1;
			 // otherwise we can't find anything steeper than the current min_ratio
			 // (see min_ratio assignment)
			 r * r <= (1.0f - h) * (1.0f - h) * ratio2 * max(width, height) *
										max(width, height);
			 ++r) {

		// Right side

		cu = u + r; // u displacement
		// set v limits
		start = v - r;
		end = v + r - 1;
		for (int cv = start; cv < end; ++cv) { // go through side
			// check if local maxima in the given direction
			int discrete_dir = 0;
			if (!(local_max_4dirs[cv * width + cu] & 1 << discrete_dir))
				continue; // skip if not
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}

		// Top side

		cv = v - r; // v displacement
		// set u limits
		start = u - r;
		end = u + r - 1;
		for (int cu = start; cu < end; ++cu) { // go through side
			// check if local maxima in the given direction
			int discrete_dir = 1;
			if (!(local_max_4dirs[cv * width + cu] & 1 << discrete_dir))
				continue; // skip if not
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}

		// Left side

		cu = u - r; // u displacement
		// set v limits
		start = v - r + 1;
		end = v + r;
		for (int cv = start; cv < end; ++cv) { // go through side
			// check if local maxima in the given direction
			int discrete_dir = 2;
			if (!(local_max_4dirs[cv * width + cu] & 1 << discrete_dir))
				continue; // skip if not
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}

		// Bottom side

		cv = v + r; // v displacement
		// set u limits
		start = u - r + 1;
		end = u + r;
		for (int cu = start; cu < end; ++cu) { // go through side
			// check if local maxima in the given direction
			int discrete_dir = 3;
			if (!(local_max_4dirs[cv * width + cu] & 1 << discrete_dir))
				continue; // skip if not
			limit_cone(heightmap, width, height, u, v, cu, cv, iwidth, iheight, h,
									 ratio2);
		}
	}

	// Square root of cone ratio -> must be sqared in rendering shader
	float sqrt_ratio = sqrt(sqrt(ratio2));


	// First order derivatives
	const int2 sums = sobel(heightmap, u, v, width, height);
	const uint8_t dhdu = (sums.x + 1020) / 8;
	const uint8_t dhdv = (sums.y + 1020) / 8;

	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<uint8_t>(sqrt_ratio * 255);
	cone_map[idx * 4 + 2] = dhdu;
	cone_map[idx * 4 + 3] = dhdv;
}

//Shared memory
__global__ void create_cone_map_analytic_shared_mem(const uint8_t* __restrict__ heightmap,
																										const bool* __restrict__ watershed,
																										const float* __restrict__ fod_dirs,
																										const int* __restrict__ fods,
																										uint8_t* __restrict__ cone_map,
																										const int width, const int height)
{
	__shared__ uint8_t s_heightmap[64];
	__shared__ bool s_watershed[64];
	__shared__ float s_fod_dirs[64];
	__shared__ unsigned int block_finished_flags;

	// global indices
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = v * width + u;

	// local indices
	const int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
	const int lane_idx = local_idx & 31;
	const int warp_idx = local_idx >> 5;

	// variables for cone setting
	const float iwidth = 1.0f / width;
	const float iheight = 1.0f / height;
	const float side = max(width, height);
	const float h = heightmap[idx] / 255.0f; // normalized height
	float ratio2 = 1.0f; // squared tangent of half aperture angle

	// finished flags
	bool finished = false;
	if (u >= width || v >= height)
		finished = true;

	if (local_idx == 0)
		block_finished_flags = 0;

	__syncthreads();

	// radius search variables
	int r = 0;
	int step = 0;

	while (block_finished_flags != 3) {
		// Get block and increment step along search layer
		int2 bidx = get_search_block_idx(r, step++);

		__syncthreads();

		// All threads copy to shared memory
		s_heightmap[local_idx] = heightmap[index(width, height, bidx.x * 8 + threadIdx.x, bidx.y * 8 + threadIdx.y)];
		s_watershed[local_idx] = watershed[index(width, height, bidx.x * 8 + threadIdx.x, bidx.y * 8 + threadIdx.y)];
		s_fod_dirs[local_idx] = fod_dirs[index(width, height, bidx.x * 8 + threadIdx.x, bidx.y * 8 + threadIdx.y)];
		__syncthreads();

		// Active threads go through block and update their cones
		for (int k = 0; k < 64; ++k) {
			limit_cone(s_heightmap[k], s_watershed[k], s_fod_dirs[k], u, v, iwidth, iheight, bidx.x, bidx.y, k, h, ratio2);
		}

		// If the search in the current radius is complete, start the next one if needed
		if (step >= 8 * r) {
			++r; step = 0; // next ring

			// check if the next is too far away for any contribution
			finished = r * r * 64 > (1.0f - h) * (1.0f - h) * ratio2 * side * side;
			
			bool warp_finished = __all_sync(0xffffffff, finished); // warp-level check
			if (lane_idx == 0 && warp_finished) // one thread per warp communicates
				atomicOr(&block_finished_flags, 1u << warp_idx);
			__syncthreads();
		}
	}

	// Square root of cone ratio -> must be sqared in rendering shader
	float sqrt_ratio = sqrt(sqrt(ratio2));

	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<uint8_t>(sqrt_ratio * 255);
	cone_map[idx * 4 + 2] = (fods[idx * 2] + 1020) / 8;
	cone_map[idx * 4 + 3] = (fods[idx * 2 + 1] + 1020) / 8;
}

__global__ void create_cone_map_8dir_shared_mem(const uint8_t* __restrict__ heightmap,
																								const uint8_t* __restrict__ local_max_8dirs,
																								uint8_t* __restrict__ cone_map,
																								const int width, const int height)
{
	__shared__ uint8_t s_heightmap[64];
	__shared__ uint8_t s_local_max_8dirs[64];
	__shared__ unsigned int block_finished_flags;

	// global indices
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = v * width + u;

	// local indices
	const int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
	const int lane_idx = local_idx & 31;
	const int warp_idx = local_idx >> 5;

	// variables for cone setting
	const float iwidth = 1.0f / width;
	const float iheight = 1.0f / height;
	const float side = max(width, height);
	const float h = heightmap[idx] / 255.0f; // normalized height
	float ratio2 = 1.0f; // squared tangent of half aperture angle

	// finished flags
	bool finished = false;
	if (u >= width || v >= height)
		finished = true;

	if (local_idx == 0)
		block_finished_flags = 0;

	__syncthreads();

	// radius search variables
	int r = 0;
	int step = 0;

	while (block_finished_flags != 3) {
		// Get block and increment step along search layer
		int2 bidx = get_search_block_idx(r, step++);

		__syncthreads();

		// All threads copy to shared memory
		s_heightmap[local_idx] = heightmap[index(width, height, bidx.x * 8 + threadIdx.x, bidx.y * 8 + threadIdx.y)];
		s_local_max_8dirs[local_idx] = local_max_8dirs[index(width, height, bidx.x * 8 + threadIdx.x, bidx.y * 8 + threadIdx.y)];
		__syncthreads();

		// Active threads go through block and update their cones
		for (int k = 0; k < 64; ++k) {
			limit_cone_8dirs(s_heightmap[k], s_local_max_8dirs[k], u, v, iwidth, iheight, bidx.x, bidx.y, k, h, ratio2);
		}

		// If the search in the current radius is complete, start the next one if needed
		if (step >= 8 * r) {
			++r; step = 0; // next ring

			// check if the next is too far away for any contribution
			finished = r * r * 64 > (1.0f - h) * (1.0f - h) * ratio2 * side * side;
			
			bool warp_finished = __all_sync(0xffffffff, finished); // warp-level check
			if (lane_idx == 0 && warp_finished) // one thread per warp communicates
				atomicOr(&block_finished_flags, 1u << warp_idx);
			__syncthreads();
		}
	}

	// Square root of cone ratio -> must be sqared in rendering shader
	float sqrt_ratio = sqrt(sqrt(ratio2));


	// First order derivatives
	int2 sums = sobel(heightmap, u, v, width, height);
	uint8_t dhdu = (sums.x + 1020) / 8;
	uint8_t dhdv = (sums.y + 1020) / 8;

	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<uint8_t>(sqrt_ratio * 255);
	cone_map[idx * 4 + 2] = dhdu;
	cone_map[idx * 4 + 3] = dhdv;
}

__global__ void create_cone_map_4dir_shared_mem(const uint8_t* __restrict__ heightmap,
																								const uint8_t* __restrict__ local_max_4dirs,
																								uint8_t* __restrict__ cone_map,
																								const int width, const int height)
{
	__shared__ uint8_t s_heightmap[64];
	__shared__ uint8_t s_local_max_4dirs[64];
	__shared__ unsigned int block_finished_flags;

	// global indices
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = v * width + u;

	// local indices
	const int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
	const int lane_idx = local_idx & 31;
	const int warp_idx = local_idx >> 5;

	// variables for cone setting
	const float iwidth = 1.0f / width;
	const float iheight = 1.0f / height;
	const float side = max(width, height);
	const float h = heightmap[idx] / 255.0f; // normalized height
	float ratio2 = 1.0f; // squared tangent of half aperture angle

	// finished flags
	bool finished = false;
	if (u >= width || v >= height)
		finished = true;

	if (local_idx == 0)
		block_finished_flags = 0;

	__syncthreads();

	// radius search variables
	int r = 0;
	int step = 0;

	while (block_finished_flags != 3) {
		// Get block and increment step along search layer
		int2 bidx = get_search_block_idx(r, step++);

		__syncthreads();

		// All threads copy to shared memory
		s_heightmap[local_idx] = heightmap[index(width, height, bidx.x * 8 + threadIdx.x, bidx.y * 8 + threadIdx.y)];
		s_local_max_4dirs[local_idx] = local_max_4dirs[index(width, height, bidx.x * 8 + threadIdx.x, bidx.y * 8 + threadIdx.y)];
		__syncthreads();

		// Active threads go through block and update their cones
		for (int k = 0; k < 64; ++k) {
			limit_cone_4dirs(s_heightmap[k], s_local_max_4dirs[k], u, v, iwidth, iheight, bidx.x, bidx.y, k, h, ratio2);
		}

		// If the search in the current radius is complete, start the next one if needed
		if (step >= 8 * r) {
			++r; step = 0; // next ring

			// check if the next is too far away for any contribution
			finished = r * r * 64 > (1.0f - h) * (1.0f - h) * ratio2 * side * side;
			
			bool warp_finished = __all_sync(0xffffffff, finished); // warp-level check
			if (lane_idx == 0 && warp_finished) // one thread per warp communicates
				atomicOr(&block_finished_flags, 1u << warp_idx);
			__syncthreads();
		}
	}

	// Square root of cone ratio -> must be sqared in rendering shader
	float sqrt_ratio = sqrt(sqrt(ratio2));


	// First order derivatives
	int2 sums = sobel(heightmap, u, v, width, height);
	uint8_t dhdu = (sums.x + 1020) / 8;
	uint8_t dhdv = (sums.y + 1020) / 8;

	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<uint8_t>(sqrt_ratio * 255);
	cone_map[idx * 4 + 2] = dhdu;
	cone_map[idx * 4 + 3] = dhdv;
}

//Packed data
__global__ void create_cone_map_analytic_packed(const uint8_t* __restrict__ heightmap,
																								const AnalyticData* __restrict__ packed,
																								const int* __restrict__ fods,
																								uint8_t* __restrict__ cone_map,
																								const int width, const int height)
{
	__shared__ AnalyticData s_packed[64];
	__shared__ unsigned int block_finished_flags;

	// global indices
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = v * width + u;

	// local indices
	const int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
	const int lane_idx = local_idx & 31;
	const int warp_idx = local_idx >> 5;

	// variables for cone setting
	const float iwidth = 1.0f / width;
	const float iheight = 1.0f / height;
	const float side = max(width, height);
	const float h = heightmap[idx] / 255.0f; // normalized height
	float ratio2 = 1.0f; // squared tangent of half aperture angle

	// finished flags
	bool finished = false;
	if (u >= width || v >= height)
		finished = true;

	if (local_idx == 0)
		block_finished_flags = 0;
	__syncthreads();

	// radius search variables
	int r = 0;
	int step = 0;

	while (block_finished_flags != 3) {
		// Get block and increment step along search layer
		int2 bidx = get_search_block_idx(r, step++);
		
		__syncthreads();

		// All threads copy to shared memory
		s_packed[local_idx] = packed[index(width, height, bidx.x * 8 + threadIdx.x, bidx.y * 8 + threadIdx.y)];
		__syncthreads();

		// Active threads go through block and update their cones
		for (int k = 0; k < 64; ++k) {
			limit_cone(s_packed[k], u, v, iwidth, iheight, bidx.x, bidx.y, k, h, ratio2);
		}

		// If the search in the current radius is complete, start the next one if needed
		if (step >= 8 * r) {
			++r; step = 0; // next ring

			// check if the next is too far away for any contribution
			finished = r * r * 64 > (1.0f - h) * (1.0f - h) * ratio2 * side * side;
			
			bool warp_finished = __all_sync(0xffffffff, finished); // warp-level check
			if (lane_idx == 0 && warp_finished) // one thread per warp communicates
				atomicOr(&block_finished_flags, 1u << warp_idx);
			__syncthreads();
		}
	}

	// Square root of cone ratio -> must be sqared in rendering shader
	float sqrt_ratio = sqrt(sqrt(ratio2));

	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<uint8_t>(sqrt_ratio * 255);
	cone_map[idx * 4 + 2] = (fods[idx * 2] + 1020) / 8;
	cone_map[idx * 4 + 3] = (fods[idx * 2 + 1] + 1020) / 8;
}

template<typename Data>
__global__ void create_cone_map_discrete_packed(const uint8_t* __restrict__ heightmap,
																								const Data* __restrict__ packed,
																								uint8_t* __restrict__ cone_map,
																								const int width, const int height)
{
	__shared__ Data s_packed[64];
	__shared__ unsigned int block_finished_flags;

	// global indices
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = v * width + u;

	// local indices
	const int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
	const int lane_idx = local_idx & 31;
	const int warp_idx = local_idx >> 5;

	// variables for cone setting
	const float iwidth = 1.0f / width;
	const float iheight = 1.0f / height;
	const float side = max(width, height);
	const float h = heightmap[idx] / 255.0f; // normalized height
	float ratio2 = 1.0f; // squared tangent of half aperture angle

	// finished flags
	bool finished = false;
	if (u >= width || v >= height)
		finished = true;

	if (local_idx == 0)
		block_finished_flags = 0;
	__syncthreads();

	// radius search variables
	int r = 0;
	int step = 0;

	while (block_finished_flags != 3) {
		// Get block and increment step along search layer
		int2 bidx = get_search_block_idx(r, step++);

		__syncthreads();

		// All threads copy to shared memory
		s_packed[local_idx] = packed[index(width, height, bidx.x * 8 + threadIdx.x, bidx.y * 8 + threadIdx.y)];
		__syncthreads();

		// Active threads go through block and update their cones
		for (int k = 0; k < 64; ++k) {
			limit_cone(s_packed[k], u, v, iwidth, iheight, bidx.x, bidx.y, k, h, ratio2);
		}

		// If the search in the current radius is complete, start the next one if needed
		if (step >= 8 * r) {
			++r; step = 0; // next ring

			// check if the next is too far away for any contribution
			finished = r * r * 64 > (1.0f - h) * (1.0f - h) * ratio2 * side * side;
			
			bool warp_finished = __all_sync(0xffffffff, finished); // warp-level check
			if (lane_idx == 0 && warp_finished) // one thread per warp communicates
				atomicOr(&block_finished_flags, 1u << warp_idx);
			__syncthreads();
		}
	}

	// Square root of cone ratio -> must be sqared in rendering shader
	float sqrt_ratio = sqrt(sqrt(ratio2));

	// First order derivatives
	int2 sums = sobel(heightmap, u, v, width, height);
	uint8_t dhdu = (sums.x + 1020) / 8;
	uint8_t dhdv = (sums.y + 1020) / 8;

	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<uint8_t>(sqrt_ratio * 255);
	cone_map[idx * 4 + 2] = dhdu;
	cone_map[idx * 4 + 3] = dhdv;
}

//Continuously packed data
__global__ void create_cone_map_analytic_continuous(const uint8_t* __restrict__ heightmap,
																										const IndexedAnalyticData* __restrict__ packed,
																										const int* __restrict__ fods,
																										uint8_t* __restrict__ cone_map,
																										const int width, const int height)
{
	__shared__ IndexedAnalyticData s_packed[64];
	__shared__ unsigned int block_finished_flags;

	// global indices
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = v * width + u;

	// local indices
	const int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
	const int lane_idx = local_idx & 31;
	const int warp_idx = local_idx >> 5;

	// variables for cone setting
	const float iwidth = 1.0f / width;
	const float iheight = 1.0f / height;
	const float side = max(width, height);
	const float h = heightmap[idx] / 255.0f; // normalized height
	float ratio2 = 1.0f; // squared tangent of half aperture angle

	// finished flags
	bool finished = false;
	if (u >= width || v >= height)
		finished = true;

	if (local_idx == 0)
		block_finished_flags = 0;
	__syncthreads();

	// radius search variables
	int r = 0;
	int step = 0;

	while (block_finished_flags != 3) {
		// Get block and increment step along search layer
		int2 bidx = get_search_block_idx(r, step++);
		
		__syncthreads();

		// All threads copy to shared memory
		s_packed[local_idx] = packed[index(width, height, bidx.x * 8 + threadIdx.x, bidx.y * 8 + threadIdx.y)];
		__syncthreads();

		// Active threads go through block and update their cones
		for (int k = 0; k < 64 && not_empty(s_packed[k]); ++k) {
			limit_cone(s_packed[k], u, v, iwidth, iheight, bidx.x, bidx.y, h, ratio2);
		}

		// If the search in the current radius is complete, start the next one if needed
		if (step >= 8 * r) {
			++r; step = 0; // next ring

			// check if the next is too far away for any contribution
			finished = r * r * 64 > (1.0f - h) * (1.0f - h) * ratio2 * side * side;
			
			bool warp_finished = __all_sync(0xffffffff, finished); // warp-level check
			if (lane_idx == 0 && warp_finished) // one thread per warp communicates
				atomicOr(&block_finished_flags, 1u << warp_idx);
			__syncthreads();
		}
	}

	// Square root of cone ratio -> must be sqared in rendering shader
	float sqrt_ratio = sqrt(sqrt(ratio2));

	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<uint8_t>(sqrt_ratio * 255);
	cone_map[idx * 4 + 2] = (fods[idx * 2] + 1020) / 8;
	cone_map[idx * 4 + 3] = (fods[idx * 2 + 1] + 1020) / 8;
}

template<typename Data>
__global__ void create_cone_map_discrete_continuous(const uint8_t* __restrict__ heightmap,
																										const Data* __restrict__ packed,
																										uint8_t* __restrict__ cone_map,
																										const int width, const int height)
{
	__shared__ Data s_packed[64];
	__shared__ unsigned int block_finished_flags;

	// global indices
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = v * width + u;

	// local indices
	const int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
	const int lane_idx = local_idx & 31;
	const int warp_idx = local_idx >> 5;

	// variables for cone setting
	const float iwidth = 1.0f / width;
	const float iheight = 1.0f / height;
	const float side = max(width, height);
	const float h = heightmap[idx] / 255.0f; // normalized height
	float ratio2 = 1.0f; // squared tangent of half aperture angle

	// finished flags
	bool finished = false;
	if (u >= width || v >= height)
		finished = true;

	if (local_idx == 0)
		block_finished_flags = 0;
	__syncthreads();

	// radius search variables
	int r = 0;
	int step = 0;

	while (block_finished_flags != 3) {
		// Get block and increment step along search layer
		int2 bidx = get_search_block_idx(r, step++);

		__syncthreads();

		// All threads copy to shared memory
		s_packed[local_idx] = packed[index(width, height, bidx.x * 8 + threadIdx.x, bidx.y * 8 + threadIdx.y)];
		__syncthreads();

		// Active threads go through block and update their cones
		for (int k = 0; k < 64 && not_empty(s_packed[k]); ++k) {
			limit_cone(s_packed[k], u, v, iwidth, iheight, bidx.x, bidx.y, h, ratio2);
		}

		// If the search in the current radius is complete, start the next one if needed
		if (step >= 8 * r) {
			++r; step = 0; // next ring

			// check if the next is too far away for any contribution
			finished = r * r * 64 > (1.0f - h) * (1.0f - h) * ratio2 * side * side;
			
			bool warp_finished = __all_sync(0xffffffff, finished); // warp-level check
			if (lane_idx == 0 && warp_finished) // one thread per warp communicates
				atomicOr(&block_finished_flags, 1u << warp_idx);
			__syncthreads();
		}
	}

	// Square root of cone ratio -> must be sqared in rendering shader
	float sqrt_ratio = sqrt(sqrt(ratio2));

	// First order derivatives
	int2 sums = sobel(heightmap, u, v, width, height);
	uint8_t dhdu = (sums.x + 1020) / 8;
	uint8_t dhdv = (sums.y + 1020) / 8;

	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<uint8_t>(sqrt_ratio * 255);
	cone_map[idx * 4 + 2] = dhdu;
	cone_map[idx * 4 + 3] = dhdv;
}
