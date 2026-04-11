// STD
#include <cmath>

// CUDA
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdint>

#include "kernels.cuh"

/* Utils */

struct Data {
	uint8_t x;		// 3 bits
	uint8_t y;		// 3 bits
	uint8_t h;		// 6 bits
	uint8_t dirs; // 4 bit mask
};

__device__ __forceinline__ Data decompress(uint16_t compressed) {
	return Data{uint8_t(compressed >> 13),
							uint8_t((compressed >> 10) & 0x7),
							uint8_t((compressed >> 4) & 0x3F),
							uint8_t(compressed & 0xF)};
}

__device__ __forceinline__ uint16_t compress(const Data &d) {
	return ((d.x & 0x7) << 13) |
				 ((d.y & 0x7) << 10) |
				 ((d.h & 0x3F) << 4) |
				  (d.dirs & 0xF);
}

__device__ __forceinline__ int dir4(int du, int dv) {
	if (abs(du) > abs(dv))
		return du > 0 ? 0 : 2; // E / W
	else
		return dv > 0 ? 1 : 3; // N / S
}

__device__ __forceinline__ int dir8(int du, int dv) {
	const float t = 0.41421356237f;

	float adu = static_cast<float>(abs(du));
	float adv = static_cast<float>(abs(dv));

	return (adv <= adu * t)		? (du > 0 ? 0 : 4)
				 : (adu <= adv * t) ? (dv > 0 ? 2 : 6)
				 : (du > 0)					? (dv > 0 ? 1 : 7)
														: (dv > 0 ? 3 : 5);
}

__device__ __forceinline__ int index(int width, int height, int u, int v) {
	return (v % height + height) % height * width + (u % width + width) % width;
}

__global__ void compress(const uint8_t *__restrict__ heightmap,
												 uint16_t *__restrict__ compressed,
												 const int width, const int height) {
	const int u = blockIdx.x * blockDim.x + threadIdx.x;
	const int v = blockIdx.y * blockDim.y + threadIdx.y;
	if (u >= width || v >= height)
		return;

	const int idx = v * width + u;

	// Direction vectors: Right, Up, Left, Down
	const int du[4] = {1, 0, -1, 0};
	const int dv[4] = {0, -1, 0, 1};

	const int h = heightmap[idx];
	uint8_t dirs = 0;
	for (int dir = 0; dir < 4; ++dir) {
		if (h >  heightmap[index(width, height, u + du[dir], v + dv[dir])] &&
				h >= heightmap[index(width, height, u - du[dir], v - dv[dir])]) {
			dirs |= (1 << dir);
		}
	}

	compressed[idx] = ((threadIdx.x & 0x7) << 13) |
								((threadIdx.y & 0x7) << 10) |
								((h           & 0xFC) << 2) |
								 (dirs        & 0xF);
}

__global__ void compress_continuously(const uint8_t *__restrict__ heightmap,
																	uint16_t *__restrict__ compressed,
																	const int width, const int height) {
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
	uint16_t compressed_value = 0;

	if (u < width && v < height) { // if in bounds check dirs
		// Direction vectors: Right, Up, Left, Down
		const int du[4] = {1, 0, -1, 0};
		const int dv[4] = {0, -1, 0, 1};

		const uint8_t h = heightmap[idx];
		uint8_t dirs = 0;
		for (int dir = 0; dir < 4; ++dir) {
			if (h >  heightmap[index(width, height, u + du[dir], v + dv[dir])] &&
					h >= heightmap[index(width, height, u - du[dir], v - dv[dir])]) {
				dirs |= (1 << dir);
			}
		}

		limiting = dirs != 0; // limiting point in any direction
		compressed_value = ((threadIdx.x & 0x7) << 13) |
									 ((threadIdx.y & 0x7) << 10) |
									 ((h           & 0xFC) << 2) |
									 (dirs         & 0xF);
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
		compressed[out_idx] = compressed_value;
	}

	// Set terminating null value
	if (local_idx == 32) { // first thread of second warp
		int block_count = warp0_count + warp_count;
		if (block_count < 64) {
			int out_idx = block_base + (block_count / blockDim.x) * width + (block_count % blockDim.x);
			compressed[out_idx] = 0;
		}
	}
}

//TODO pack continuously

__global__ void invert(uint8_t *data, const int width, const int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height)
		return;

	data[v * width + u] = 255 - data[v * width + u];
}

__global__ void bits_to_image(const uint8_t *__restrict__ data,
															uint8_t *__restrict__ output_image,
															const int width, const int height,
															const uint8_t bitmask) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height)
		return;

	output_image[v * width + u] = data[v * width + u] & bitmask ? 255 : 0;
}

/* Derivatives */

__device__ __constant__ float hkernel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

__device__ __constant__ float vkernel[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

__global__ void fod(const uint8_t *__restrict__ heightmap,
										int *__restrict__ fods,
										float *__restrict__ fod_dirs,
										const int width, const int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height)
		return;

	int idx = v * width + u;

	int hsum = 0;
	int vsum = 0;
	for (int dv = 0; dv < 3; ++dv) {
		for (int du = 0; du < 3; ++du) {
			int didx = index(width, height, u + du - 1, v + dv - 1);
			hsum += heightmap[didx] * hkernel[dv * 3 + du];
			vsum += heightmap[didx] * vkernel[dv * 3 + du];
		}
	}

	fod_dirs[idx] = atan2f(vsum, hsum);

	fods[idx * 2 + 0] = hsum;
	fods[idx * 2 + 1] = vsum;
}

__global__ void watershed(const int *__restrict__ fods,
													bool *__restrict__ watersheds,
													const int width, const int height) {
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

/* Local maxima */

__global__ void non_maximum_suppression(const uint8_t *__restrict__ heightmap,
																				const float *__restrict__ fod_dirs,
																				const bool *watershed,
																				bool *__restrict__ suppressed,
																				const int width, const int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height)
		return;

	int idx = v * width + u;

	if (!watershed[idx]) {
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
		suppressed[idx] = watershed[idx];
}

__global__ void local_max_8dir(const uint8_t *__restrict__ heightmap,
															 uint8_t *__restrict__ local_max_8dirs,
															 const int width, const int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height)
		return;

	int h = heightmap[v * width + u];

	// Direction vectors: Right, Up-Right, Up, Up-Left, Left, Down-Left, Down,
	// Down-Right
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
	// TODO save difference as metric

	local_max_8dirs[v * width + u] = result;
}

__global__ void local_max_4dir(const uint8_t *__restrict__ heightmap,
															 uint8_t *__restrict__ local_max_4dirs,
															 const int width, const int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height)
		return;

	int idx = v * width + u;
	int h = heightmap[idx];

	// Direction vectors: Right, Up, Left, Down
	const int du[4] = {1, 0, -1, 0};
	const int dv[4] = {0, -1, 0, 1};

	uint8_t result = 0;

#pragma unroll
	for (int dir = 0; dir < 4; ++dir) {
		if (h > heightmap[index(width, height, u + du[dir], v + dv[dir])] &&
				h >= heightmap[index(width, height, u - du[dir], v - dv[dir])]) {
			result |= (1 << dir);
		}
	}

	local_max_4dirs[idx] = result;
}

/* Cone maps */

__device__ void limit_cone(const uint8_t *heightmap,
													 const int width, const int height,
													 const int u, const int v, const int du, const int dv,
													 const float iwidth, const float iheight,
													 const float h, float &ratio2) {
	// normalize u and v displacements
	float dun = (du - u) * iwidth;
	float dvn = (dv - v) * iheight;

	float d2 = dun * dun + dvn * dvn; // distance squared

	// height difference
	float dh = heightmap[index(width, height, du, dv)] / 255.0 - h;

	// if more steep than previous best, override
	if (dh > 0.0f && dh * dh * ratio2 > d2)
		ratio2 = d2 / (dh * dh);
}

//TODO
__global__ void create_cone_map_analytic_local_mem(const uint8_t *__restrict__ heightmap,
																									 const bool *__restrict__ suppressed,
																									 const float *__restrict__ fod_dirs,
																									 const int *__restrict__ fods,
																									 uint8_t *__restrict__ cone_map,
																									 const int width, const int height) {
	__shared__ uint8_t l_heightmap[64];
	__shared__ bool l_suppressed[64];
	__shared__ float l_fod_dirs[64];
	__shared__ unsigned int block_finished_flags;

	// global indices
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = v * width + u;

	// local indices
	int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
	int lane_idx = local_idx & 31;
	int warp_idx = local_idx >> 5;

	// finished flags
	bool finished = false;
	if (u >= width || v >= height)
		finished = true;

	if (local_idx == 0)
		block_finished_flags = 0;

	// variables for cone setting
	float iwidth = 1.0f / width;
	float iheight = 1.0f / height;
	float ratio2 = 1.0f;							 // squared tangent of half aperture angle
	float h = heightmap[idx] / 255.0f; // normalized height

	// radius search variables
	int r = 0;
	int step = 0;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	__syncthreads();

	do {
		// All threads copy to shared memory
		l_heightmap[local_idx] = heightmap[index(
				width, height, bx * 8 + threadIdx.x, by * 8 + threadIdx.y)];
		l_suppressed[local_idx] = suppressed[index(
				width, height, bx * 8 + threadIdx.x, by * 8 + threadIdx.y)];
		l_fod_dirs[local_idx] = fod_dirs[index(width, height, bx * 8 + threadIdx.x,
																					 by * 8 + threadIdx.y)];

		__syncthreads();

		// Active threads update their cones
		if (!finished) {
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j) {
					int du = bx * 8 + j - u;
					int dv = by * 8 + i - v;

					if (!l_suppressed[i * 8 + j] ||
							abs(remainderf((l_fod_dirs[i * 8 + j] - atan2f(dv, du) - M_PI_2),
														 M_PI)) > (M_PI / 8.0f))
						continue;

					// normalize u and v displacements
					float dun = du * iwidth;
					float dvn = dv * iheight;

					float d2 = dun * dun + dvn * dvn; // distance squared

					// height difference
					float dh = l_heightmap[i * 8 + j] / 255.0f - h;

					// if more steep than previous best, override
					if (dh > 0.0f && dh * dh * ratio2 > d2)
						ratio2 = d2 / (dh * dh);
				}
			}
		}

		// increase radius if the previous layer has been finished
		if (step >= 8 * r) {
			++r; // next ring
			step = 0;

			// check if the next is too far away for any contribution
			if (r * r * 64 > (1.0f - h) * (1.0f - h) * ratio2 * max(width, height) *
													 max(width, height))
				finished = true;

			// warp-level check
			bool warp_finished = __all_sync(0xffffffff, finished);

			// communicate between warps
			if (lane_idx == 0 && warp_finished)
				atomicOr(&block_finished_flags, 1u << warp_idx);

			__syncthreads();
		}

		// step along layer
		if (step < 2 * r) {
			// Right side:
			bx = blockIdx.x + r;
			by = blockIdx.y - r + step;
		} else if (step < 4 * r) {
			// Top side:
			bx = blockIdx.x - r + (step - 2 * r);
			by = blockIdx.y - r;
		} else if (step < 6 * r) {
			// Left side:
			bx = blockIdx.x - r;
			by = blockIdx.y + r - (step - 4 * r);
		} else {
			// Bottom side:
			bx = blockIdx.x + r - (step - 6 * r);
			by = blockIdx.y + r;
		}
		++step;

	} while (block_finished_flags != 3);

	float ratio = sqrt(ratio2);
	// most of the data is on the low end...sqrting again spreads it better
	// (plus multiply is a cheap operation in shaders!)
	// -- Dummer
	ratio = sqrt(ratio);
	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<uint8_t>(ratio * 255);
	cone_map[idx * 4 + 2] = (fods[idx * 2] + 1020) / 8;
	cone_map[idx * 4 + 3] = (fods[idx * 2 + 1] + 1020) / 8;
}

//TODO
__global__ void create_cone_map_analytic(const uint8_t *__restrict__ heightmap,
																				 const bool *__restrict__ suppressed,
																				 const float *__restrict__ fod_dirs,
																				 const int *__restrict__ fods,
																				 uint8_t *__restrict__ cone_map,
																				 const int width, const int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height)
		return;

	int idx = v * width + u;

	float iwidth = 1.0f / width;
	float iheight = 1.0f / height;

	float ratio2 = 1.0f;

	// normalize height
	float h = heightmap[idx] / 255.0f;

	// init variables
	int du, dv;

	// search with increasing radius around the texel
	for (int r = 1;
			 // otherwise we can't find anything steeper than the current min_ratio
			 // (see min_ratio assignment)
			 r * r <= (1.0f - h) * (1.0f - h) * ratio2 * max(width, height) *
										max(width, height);
			 ++r) {

		// Right side

		// u displacement
		du = u + r;

		// go through side
		for (int dv = v - r; dv < v + r - 1; ++dv) {
			// check if (suppressed) watershed point, skip if not
			if (!suppressed[index(width, height, du, dv)] ||
					abs(remainderf((fod_dirs[index(width, height, du, dv)] -
													atan2f(dv, du) - M_PI_2),
												 M_PI)) > (M_PI / 8.0))
				continue;
			limit_cone(heightmap, width, height, u, v, du, dv, iwidth, iheight, h,
									 ratio2);
		}

		// Top side

		// u displacement
		dv = v - r;

		// go through side
		for (int du = u - r; du < u + r - 1; ++du) {
			// check if (suppressed) watershed point, skip if not
			if (!suppressed[index(width, height, du, dv)] ||
					abs(remainderf((fod_dirs[index(width, height, du, dv)] -
													atan2f(dv, du) - M_PI_2),
												 M_PI)) > (M_PI / 8.0))
				continue;
			limit_cone(heightmap, width, height, u, v, du, dv, iwidth, iheight, h,
									 ratio2);
		}

		// Left side

		// u displacement
		du = u - r;

		// go through side
		for (int dv = v - r + 1; dv < v + r; ++dv) {
			// check if (suppressed) watershed point, skip if not
			if (!suppressed[index(width, height, du, dv)] ||
					abs(remainderf((fod_dirs[index(width, height, du, dv)] -
													atan2f(dv, du) - M_PI_2),
												 M_PI)) > (M_PI / 8.0))
				continue;
			limit_cone(heightmap, width, height, u, v, du, dv, iwidth, iheight, h,
									 ratio2);
		}

		// Bottom side

		// u displacement
		dv = v + r;

		// go through side
		for (int du = u - r + 1; du < u + r; ++du) {
			// check if (suppressed) watershed point, skip if not
			if (!suppressed[index(width, height, du, dv)] ||
					abs(remainderf((fod_dirs[index(width, height, du, dv)] -
													atan2f(dv, du) - M_PI_2),
												 M_PI)) > (M_PI / 8.0))
				continue;
			limit_cone(heightmap, width, height, u, v, du, dv, iwidth, iheight, h,
									 ratio2);
		}
	}

	float ratio = sqrt(ratio2);
	// most of the data is on the low end...sqrting again spreads it better
	// (plus multiply is a cheap operation in shaders!)
	// -- Dummer
	ratio = sqrt(ratio);
	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<uint8_t>(ratio * 255);
	cone_map[idx * 4 + 2] = (fods[idx * 2] + 1020) / 8;
	cone_map[idx * 4 + 3] = (fods[idx * 2 + 1] + 1020) / 8;
}

//TODO
__global__ void create_cone_map_8dir_local_mem(const uint8_t *__restrict__ heightmap,
																							 const uint8_t *__restrict__ local_max_8dirs,
																							 uint8_t *__restrict__ cone_map,
																							 const int width, const int height) {
	__shared__ uint8_t l_heightmap[64];
	__shared__ uint8_t l_local_max_8dirs[64];
	__shared__ unsigned int block_finished_flags;

	// global indices
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = v * width + u;

	// local indices
	int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
	int lane_idx = local_idx & 31;
	int warp_idx = local_idx >> 5;

	// finished flags
	bool finished = false;
	if (u >= width || v >= height)
		finished = true;

	if (local_idx == 0)
		block_finished_flags = 0;

	// variables for cone setting
	float iwidth = 1.0f / width;
	float iheight = 1.0f / height;
	float ratio2 = 1.0f;							 // squared tangent of half aperture angle
	float h = heightmap[idx] / 255.0f; // normalized height

	// radius search variables
	int r = 0;
	int step = 0;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	__syncthreads();

	do {
		// All threads copy to shared memory
		l_heightmap[local_idx] = heightmap[index(
				width, height, bx * 8 + threadIdx.x, by * 8 + threadIdx.y)];
		l_local_max_8dirs[local_idx] = local_max_8dirs[index(
				width, height, bx * 8 + threadIdx.x, by * 8 + threadIdx.y)];

		__syncthreads();

		// Active threads update their cones
		if (!finished) {
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j) {
					int du = bx * 8 + j - u;
					int dv = by * 8 + i - v;

					int discrete_dir = dir8(du, dv);
					// TODO the first block separately, then only the side the block is on
					// (?)

					if (!(l_local_max_8dirs[i * 8 + j] & 1 << discrete_dir))
						continue;

					// normalize u and v displacements
					float dun = du * iwidth;
					float dvn = dv * iheight;

					float d2 = dun * dun + dvn * dvn; // distance squared

					// height difference
					float dh = l_heightmap[i * 8 + j] / 255.0f - h;

					// if more steep than previous best, override
					if (dh > 0.0f && dh * dh * ratio2 > d2)
						ratio2 = d2 / (dh * dh);
				}
			}
		}

		// increase radius if the previous layer has been finished
		if (step >= 8 * r) {
			++r; // next ring
			step = 0;

			// check if the next is too far away for any contribution
			if (r * r * 64 > (1.0f - h) * (1.0f - h) * ratio2 * max(width, height) *
													 max(width, height))
				finished = true;

			// warp-level check
			bool warp_finished = __all_sync(0xffffffff, finished);

			// communicate between warps
			if (lane_idx == 0 && warp_finished)
				atomicOr(&block_finished_flags, 1u << warp_idx);

			__syncthreads();
		}

		// step along layer
		if (step < 2 * r) {
			// Right side:
			bx = blockIdx.x + r;
			by = blockIdx.y - r + step;
		} else if (step < 4 * r) {
			// Top side:
			bx = blockIdx.x - r + (step - 2 * r);
			by = blockIdx.y - r;
		} else if (step < 6 * r) {
			// Left side:
			bx = blockIdx.x - r;
			by = blockIdx.y + r - (step - 4 * r);
		} else {
			// Bottom side:
			bx = blockIdx.x + r - (step - 6 * r);
			by = blockIdx.y + r;
		}
		++step;

	} while (block_finished_flags != 3);

	float ratio = sqrt(ratio2);
	// most of the data is on the low end...sqrting again spreads it better
	// (plus multiply is a cheap operation in shaders!)
	// -- Dummer
	ratio = sqrt(ratio);

	/* First order derivative */
	int hsum = 0;
	int vsum = 0;
	for (int dv = 0; dv < 3; ++dv) {
		for (int du = 0; du < 3; ++du) {
			int didx = index(width, height, u + du - 1, v + dv - 1);
			hsum += heightmap[didx] * hkernel[dv * 3 + du];
			vsum += heightmap[didx] * vkernel[dv * 3 + du];
		}
	}

	uint8_t dhdu = (hsum + 1020) / 8;
	uint8_t dhdv = (vsum + 1020) / 8;

	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<uint8_t>(ratio * 255);
	cone_map[idx * 4 + 2] = dhdu;
	cone_map[idx * 4 + 3] = dhdv;
}

//TODO
__global__ void create_cone_map_8dir(const uint8_t *__restrict__ heightmap,
																		 const uint8_t *__restrict__ local_max_8dirs,
																		 uint8_t *__restrict__ cone_map,
																		 const int width, const int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height)
		return;

	/* First order derivative */

	int hsum = 0;
	int vsum = 0;
	for (int dv = 0; dv < 3; ++dv) {
		for (int du = 0; du < 3; ++du) {
			int didx = index(width, height, u + du - 1, v + dv - 1);
			hsum += heightmap[didx] * hkernel[dv * 3 + du];
			vsum += heightmap[didx] * vkernel[dv * 3 + du];
		}
	}

	uint8_t dhdu = (hsum + 1020) / 8;
	uint8_t dhdv = (vsum + 1020) / 8;

	/*Cone ratios*/
	int idx = v * width + u;

	float iwidth = 1.0f / width;
	float iheight = 1.0f / height;

	// normalize height
	float h = heightmap[idx] / 255.0f;

	float ratio2 = 1.0f;

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

		// u displacement
		cu = u + r;

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

		// u displacement
		cv = v - r;

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

		// u displacement
		cu = u - r;

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

		// u displacement
		cv = v + r;

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

	float ratio = sqrt(ratio2);
	// most of the data is on the low end...sqrting again spreads it better
	// (plus multiply is a cheap operation in shaders!)
	// -- Dummer
	ratio = sqrt(ratio);
	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<uint8_t>(ratio * 255);
	cone_map[idx * 4 + 2] = dhdu;
	cone_map[idx * 4 + 3] = dhdv;
}

__device__ __forceinline__ void limit_cone(const uint16_t compressed,
																					 const int u, const int v,
																					 const float iwidth, const float iheight,
																					 const int bx, const int by,
																					 const float h, float& ratio2) {
	const int du = bx * 8 + (compressed >> 13) - u;
	const int dv = by * 8 + ((compressed >> 10) & 7) - v;
	const int discrete_dir = dir4(du, dv); // direction of texel from cone apex

	if (!(compressed & (1 << discrete_dir)))
		return; // not limiting point in this direction

	// normalize u and v displacements
	const float dun = du * iwidth;
	const float dvn = dv * iheight;

	const float d2 = dun * dun + dvn * dvn; // distance squared
	const float dh = (((compressed >> 4) & 0x3F) << 2) / 252.0f - h; // height difference
	const float dh2 = dh * dh;

	if (dh > 0.0f && dh2 * ratio2 > d2) // if steeper than current
		ratio2 = d2 / dh2; // override squared ratio
}

__device__ __forceinline__ int2 get_search_block_idx(const int r, const int step) {
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

__global__ void create_cone_map_compacted(const uint8_t *__restrict__ heightmap,
																					const uint16_t *__restrict__ compressed,
																					uint8_t *__restrict__ cone_map,
																					const int width, const int height) {
	__shared__ uint16_t s_compressed[64];
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

		// All threads copy to shared memory
		s_compressed[local_idx] = compressed[index(width, height, bidx.x * 8 + threadIdx.x, bidx.y * 8 + threadIdx.y)];
		__syncthreads();

		// Active threads go through block and update their cones
		for (int k = 0; k < 64 && s_compressed[k] != 0; ++k) {
			limit_cone(s_compressed[k], u, v, iwidth, iheight, bidx.x, bidx.y, h, ratio2);
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

	// most of the data is on the low end...sqrting again spreads it better
	// (plus multiply is a cheap operation in shaders!)
	// -- Dummer
	float sqrt_ratio = sqrt(sqrt(ratio2));

	/* First order derivative */
	int hsum = 0;
	int vsum = 0;
	for (int dv = 0; dv < 3; ++dv) {
		for (int du = 0; du < 3; ++du) {
			int didx = index(width, height, u + du - 1, v + dv - 1);
			hsum += heightmap[didx] * hkernel[dv * 3 + du];
			vsum += heightmap[didx] * vkernel[dv * 3 + du];
		}
	}

	uint8_t dhdu = (hsum + 1020) / 8;
	uint8_t dhdv = (vsum + 1020) / 8;

	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<uint8_t>(sqrt_ratio * 255);
	cone_map[idx * 4 + 2] = dhdu;
	cone_map[idx * 4 + 3] = dhdv;
}
