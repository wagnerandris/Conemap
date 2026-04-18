// STD
#include <cmath>
#include <cstdint>

// CUDA
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "kernels.cuh"


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

/* Derivatives */
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


/*Local maxima */
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


/*Cone maps */
//TODO original


//Baseline
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
	int start, end;

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
	cone_map[idx * 4 + 2] = (fods[idx * 2] + 1020) / 8;
	cone_map[idx * 4 + 3] = (fods[idx * 2 + 1] + 1020) / 8;
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
