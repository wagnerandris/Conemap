// STD
#include <cmath>

// CUDA
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "kernels.cuh"


/* Utils */

__device__ int index(int width, int height, int u, int v) {
	return (v % height + height) % height * width + (u % width + width) % width;
}

__global__ void invert(unsigned char* data, int width, int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;

	data[v * width + u] = 255 - data[v * width + u];
}

__global__ void bits_to_image(unsigned char* data, unsigned char* output_image, int width, int height, unsigned char bitmask) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;

	output_image[v * width + u] = data[v * width + u] & bitmask ? 255 : 0;
}


/* Derivatives */

__device__ __constant__ float hkernel[9] = {
 -1, 0, 1,
 -2, 0, 2,
 -1, 0, 1
};

__device__ __constant__ float vkernel[9] = {
 -1,-2,-1,
	0, 0, 0,
	1, 2, 1
};

__global__ void fod(unsigned char* heightmap, int* fods, float* fod_dirs, int width, int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;
	
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

__global__ void watershed(int* fods, bool* watersheds, int width, int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;
	
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
		vvsum * fods[idx * 2 + 0] * fods[idx * 2 + 0]
		< 0;
}


/* Local maxima */

__global__ void non_maximum_suppression(unsigned char* heightmap, float* fod_dirs, bool* watershed, bool* suppressed, int width, int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;
	
	int idx = v * width + u;
	
	if (!watershed[idx]) {
		suppressed[idx] = 0;
		return;
	}

	// get the nearest discrete direction
	unsigned char ddir = static_cast<unsigned char>(
																(fod_dirs[idx] + M_PI // all positive
																+ M_PI_4f / 2.0f) // align regions
																/ M_PI_4f // 8 dirs
																) % 4; // opposite ones are the same

	// neighbours orthogonal to ddir
	const int du[4] = { 0, -1, 1, 1};
	const int dv[4] = { 1,	1, 0, 1};

	// if one of the neighbours has greater height, suppress the current texel
	if			(heightmap[index(width, height, u + du[ddir], v + dv[ddir])] > heightmap[idx]) suppressed[idx] = 0;
	else if (heightmap[index(width, height, u - du[ddir], v - dv[ddir])] > heightmap[idx]) suppressed[idx] = 0;
	else suppressed[idx] = watershed[idx];
}

__global__ void local_max_8dir(unsigned char* heightmap, unsigned char* local_max_8dirs, int width, int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;

	int h = heightmap[v * width + u];

	// Direction vectors: Right, Up-Right, Up, Up-Left, Left, Down-Left, Down, Down-Right
	const int du[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	const int dv[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	unsigned char result = 0;

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


/* Cone maps */

__device__ void limit_ratio2(unsigned char* heightmap, int width, int height, int u, int v, int du, int dv, float iwidth, float iheight, float h, float &ratio2) {
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

__global__ void create_cone_map_analytic_local_mem(unsigned char* heightmap, bool* suppressed, float* fod_dirs, int* fods, unsigned char* cone_map, int width, int height) {
	__shared__ unsigned char l_heightmap[64];
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
	if (u >= width || v >= height) finished = true;

	if (local_idx == 0)
		block_finished_flags = 0;
	__syncthreads();
	
	// variables for cone setting
	float iwidth = 1.0f / width;
	float iheight = 1.0f / height;
	float ratio2 = 1.0f; // squared tangent of half aperture angle
	float h = heightmap[idx] / 255.0f; // normalized height

	// radius search variables
	int r = 0;
	int step = 0;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	__syncthreads();

	do {
		// All threads copy to shared memory
		l_heightmap[local_idx] = heightmap[index(width, height, bx * 8 + threadIdx.x, by * 8 + threadIdx.y)];
		l_suppressed[local_idx] = suppressed[index(width, height, bx * 8 + threadIdx.x, by * 8 + threadIdx.y)];
		l_fod_dirs[local_idx] = fod_dirs[index(width, height, bx * 8 + threadIdx.x, by * 8 + threadIdx.y)];

		__syncthreads();

		// Active threads update their cones
		if (!finished) {
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j) {
					int du = bx * 8 + j - u;
					int dv = by * 8 + i - v;

					if (!l_suppressed[i * 8 + j] ||
						abs(remainderf((l_fod_dirs[i * 8 + j] - atan2f(dv, du) - M_PI_2), M_PI)) > (M_PI / 8.0f))
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
			if (r * r * 64 > (1.0f - h) * (1.0f - h) * ratio2 * max(width, height) * max(width, height))
				finished = true;

			// warp-level check
			unsigned mask = __ballot_sync(0xffffffff, finished);
			bool warp_finished = (mask == 0xffffffff);

			// communicate between warps
			if (lane_idx == 0 && warp_finished) atomicOr(&block_finished_flags, 1u << warp_idx);

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
	cone_map[idx * 4 + 1] = static_cast<unsigned char>(ratio * 255);
	cone_map[idx * 4 + 2] = (fods[idx * 2] + 1020) / 8;
	cone_map[idx * 4 + 3] = (fods[idx * 2 + 1] + 1020) / 8;
}

__global__ void create_cone_map_analytic(unsigned char* heightmap, bool* suppressed, float* fod_dirs, int* fods, unsigned char* cone_map, int width, int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;

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
			// otherwise we can't find anything steeper than the current min_ratio (see min_ratio assignment)
			r * r <= (1.0f - h) * (1.0f - h) * ratio2 * max(width, height) * max(width, height);
			++r) {

		// Right side

		// u displacement
		du = u + r;

		// go through side
		for (int dv = v - r; dv < v + r - 1; ++dv) {
			// check if (suppressed) watershed point, skip if not
			if (!suppressed[index(width, height, du, dv)] ||
					abs(remainderf((fod_dirs[index(width, height, du, dv)] - atan2f(dv, du) - M_PI_2), M_PI)) > (M_PI / 8.0))
				continue;
			limit_ratio2(heightmap, width, height, u, v, du, dv, iwidth, iheight, h, ratio2);
		}

		// Top side

		// u displacement
		dv = v - r;

		// go through side
		for (int du = u - r; du < u + r - 1; ++du) {
			// check if (suppressed) watershed point, skip if not
			if (!suppressed[index(width, height, du, dv)] ||
					abs(remainderf((fod_dirs[index(width, height, du, dv)] - atan2f(dv, du) - M_PI_2), M_PI)) > (M_PI / 8.0))
				continue;
			limit_ratio2(heightmap, width, height, u, v, du, dv, iwidth, iheight, h, ratio2);
		}

		// Left side

		// u displacement
		du = u - r;

		// go through side
		for (int dv = v - r + 1; dv < v + r; ++dv) {
			// check if (suppressed) watershed point, skip if not
			if (!suppressed[index(width, height, du, dv)] ||
					abs(remainderf((fod_dirs[index(width, height, du, dv)] - atan2f(dv, du) - M_PI_2), M_PI)) > (M_PI / 8.0))
				continue;
			limit_ratio2(heightmap, width, height, u, v, du, dv, iwidth, iheight, h, ratio2);
		}

		// Bottom side

		// u displacement
		dv = v + r;

		// go through side
		for (int du = u - r + 1; du < u + r; ++du) {
			// check if (suppressed) watershed point, skip if not
			if (!suppressed[index(width, height, du, dv)] ||
					abs(remainderf((fod_dirs[index(width, height, du, dv)] - atan2f(dv, du) - M_PI_2), M_PI)) > (M_PI / 8.0))
				continue;
			limit_ratio2(heightmap, width, height, u, v, du, dv, iwidth, iheight, h, ratio2);
		}
	}

	float ratio = sqrt(ratio2);
	// most of the data is on the low end...sqrting again spreads it better
	// (plus multiply is a cheap operation in shaders!)
	// -- Dummer
	ratio = sqrt(ratio);
	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<unsigned char>(ratio * 255);
	cone_map[idx * 4 + 2] = (fods[idx * 2] + 1020) / 8;
	cone_map[idx * 4 + 3] = (fods[idx * 2 + 1] + 1020) / 8;
}

__global__ void create_cone_map_8dir_local_mem(unsigned char* heightmap, unsigned char* local_max_8dirs, unsigned char* cone_map, int width, int height) {
	__shared__ unsigned char l_heightmap[64];
	__shared__ unsigned char l_local_max_8dirs[64];
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
	if (u >= width || v >= height) finished = true;

	if (local_idx == 0)
		block_finished_flags = 0;
	__syncthreads();
	
	// variables for cone setting
	float iwidth = 1.0f / width;
	float iheight = 1.0f / height;
	float ratio2 = 1.0f; // squared tangent of half aperture angle
	float h = heightmap[idx] / 255.0f; // normalized height

	// radius search variables
	int r = 0;
	int step = 0;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	__syncthreads();

	do {
		// All threads copy to shared memory
		l_heightmap[local_idx] = heightmap[index(width, height, bx * 8 + threadIdx.x, by * 8 + threadIdx.y)];
		l_local_max_8dirs[local_idx] = local_max_8dirs[index(width, height, bx * 8 + threadIdx.x, by * 8 + threadIdx.y)];

		__syncthreads();

		// Active threads update their cones
		if (!finished) {
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j) {
					int du = bx * 8 + j - u;
					int dv = by * 8 + i - v;

					// TODO ternary operators instead (?)
					int discrete_dir = static_cast<unsigned char>(
																(atan2f(dv, du) + M_PI // all positive
																+ M_PI_4f / 2.0f) // align regions
																/ M_PI_4f // 8 dirs
																);
					// TODO the first block separately, then only the side the block is on (?)

					if (!(l_local_max_8dirs[i * 8 + j] & 1 << discrete_dir)) continue;

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
			if (r * r * 64 > (1.0f - h) * (1.0f - h) * ratio2 * max(width, height) * max(width, height))
				finished = true;

			// warp-level check
			unsigned mask = __ballot_sync(0xffffffff, finished);
			bool warp_finished = (mask == 0xffffffff);

			// communicate between warps
			if (lane_idx == 0 && warp_finished) atomicOr(&block_finished_flags, 1u << warp_idx);

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

	unsigned char dhdu = (hsum + 1020) / 8;
	unsigned char dhdv = (vsum + 1020) / 8;

	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<unsigned char>(ratio * 255);
	cone_map[idx * 4 + 2] = dhdu;
	cone_map[idx * 4 + 3] = dhdv;
}

__global__ void create_cone_map_8dir(unsigned char* heightmap, unsigned char* local_max_8dirs, unsigned char* cone_map, int width, int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;

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

	unsigned char dhdu = (hsum + 1020) / 8;
	unsigned char dhdv = (vsum + 1020) / 8;

/*Cone ratios*/
	int idx = v * width + u;

	float iwidth = 1.0f / width;
	float iheight = 1.0f / height;

	// normalize height
	float h = heightmap[idx] / 255.0f;
	
	float ratio2 = 1.0f;

	// init variables
	int du, dv;
	int start, end;

	// search with increasing radius around the texel
	for (int r = 1;
			// otherwise we can't find anything steeper than the current min_ratio (see min_ratio assignment)
			r * r <= (1.0f - h) * (1.0f - h) * ratio2 * max(width, height) * max(width, height);
			++r) {

		// Right side

		// u displacement
		du = u + r;

		// set v limits
		start = v - r;
		end = v + r - 1;

		// go through side
		// check if local maxima in the given direction
		// skip if not
		for (int dv = start; dv <= start + r / 2; ++dv) {
			int discrete_dir = 1;
			if (!(local_max_8dirs[dv * width + du] & 1 << discrete_dir)) continue;
			limit_ratio2(heightmap, width, height, u, v, du, dv, iwidth, iheight, h, ratio2);
		}
		for (int dv = start + r / 2 + 1; dv < end - r / 2; ++dv) {
			int discrete_dir = 0;
			if (!(local_max_8dirs[dv * width + du] & 1 << discrete_dir)) continue;
			limit_ratio2(heightmap, width, height, u, v, du, dv, iwidth, iheight, h, ratio2);
		}
		for (int dv = end - r / 2; dv < end; ++dv) {
			int discrete_dir = 7;
			if (!(local_max_8dirs[dv * width + du] & 1 << discrete_dir)) continue;
			limit_ratio2(heightmap, width, height, u, v, du, dv, iwidth, iheight, h, ratio2);
		}

		// Top side

		// u displacement
		dv = v - r;

		// set u limits
		start = u - r;
		end = u + r - 1;

		// go through side
		// check if local maxima in the given direction
		// skip if not
		for (int du = start; du <= start + r / 2; ++du) {
			int discrete_dir = 3;
			if (!(local_max_8dirs[dv * width + du] & 1 << discrete_dir)) continue;
			limit_ratio2(heightmap, width, height, u, v, du, dv, iwidth, iheight, h, ratio2);
		}
		for (int du = start + r / 2 + 1; du < end - r / 2; ++du) {
			int discrete_dir = 2;
			if (!(local_max_8dirs[dv * width + du] & 1 << discrete_dir)) continue;
			limit_ratio2(heightmap, width, height, u, v, du, dv, iwidth, iheight, h, ratio2);
		}
		for (int du = end - r / 2; du < end; ++du) {
			int discrete_dir = 1;
			if (!(local_max_8dirs[dv * width + du] & 1 << discrete_dir)) continue;
			limit_ratio2(heightmap, width, height, u, v, du, dv, iwidth, iheight, h, ratio2);
		}

		// Left side

		// u displacement
		du = u - r;

		// set v limits
		start = v - r + 1;
		end = v + r;

		// go through side
		// check if local maxima in the given direction
		// skip if not
		for (int dv = start; dv <= start + r / 2; ++dv) {
			int discrete_dir = 3;
			if (!(local_max_8dirs[dv * width + du] & 1 << discrete_dir)) continue;
			limit_ratio2(heightmap, width, height, u, v, du, dv, iwidth, iheight, h, ratio2);
		}
		for (int dv = start + r / 2 + 1; dv < end - r / 2; ++dv) {
			int discrete_dir = 4;
			if (!(local_max_8dirs[dv * width + du] & 1 << discrete_dir)) continue;
			limit_ratio2(heightmap, width, height, u, v, du, dv, iwidth, iheight, h, ratio2);
		}
		for (int dv = end - r / 2; dv < end; ++dv) {
			int discrete_dir = 5;
			if (!(local_max_8dirs[dv * width + du] & 1 << discrete_dir)) continue;
			limit_ratio2(heightmap, width, height, u, v, du, dv, iwidth, iheight, h, ratio2);
		}

		// Bottom side

		// u displacement
		dv = v + r;

		// set u limits
		start = u - r + 1;
		end = u + r;

		// go through side
		// check if local maxima in the given direction
		// skip if not
		for (int du = start; du <= start + r / 2; ++du) {
			int discrete_dir = 5;
			if (!(local_max_8dirs[dv * width + du] & 1 << discrete_dir)) continue;
			limit_ratio2(heightmap, width, height, u, v, du, dv, iwidth, iheight, h, ratio2);
		}
		for (int du = start + r / 2 + 1; du < end - r / 2; ++du) {
			int discrete_dir = 6;
			if (!(local_max_8dirs[dv * width + du] & 1 << discrete_dir)) continue;
			limit_ratio2(heightmap, width, height, u, v, du, dv, iwidth, iheight, h, ratio2);
		}
		for (int du = end - r / 2; du < end; ++du) {
			int discrete_dir = 7;
			if (!(local_max_8dirs[dv * width + du] & 1 << discrete_dir)) continue;
			limit_ratio2(heightmap, width, height, u, v, du, dv, iwidth, iheight, h, ratio2);
		}
	}

	float ratio = sqrt(ratio2);
	// most of the data is on the low end...sqrting again spreads it better
	// (plus multiply is a cheap operation in shaders!)
	// -- Dummer
	ratio = sqrt(ratio);
	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<unsigned char>(ratio * 255);
	cone_map[idx * 4 + 2] = dhdu;
	cone_map[idx * 4 + 3] = dhdv;
}

__global__ void create_cone_map_4dir_local_mem(unsigned char* heightmap, unsigned char* local_max_4dirs, unsigned char* cone_map, int width, int height) {
	__shared__ unsigned char l_heightmap[64];
	__shared__ unsigned char l_local_max_4dirs[64];
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
	if (u >= width || v >= height) finished = true;

	if (local_idx == 0)
		block_finished_flags = 0;
	__syncthreads();
	
	// variables for cone setting
	float iwidth = 1.0f / width;
	float iheight = 1.0f / height;
	float ratio2 = 1.0f; // squared tangent of half aperture angle
	float h = heightmap[idx] / 255.0f; // normalized height

	// radius search variables
	int r = 0;
	int step = 0;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	__syncthreads();

	do {
		// All threads copy to shared memory
		l_heightmap[local_idx] = heightmap[index(width, height, bx * 8 + threadIdx.x, by * 8 + threadIdx.y)];
		l_local_max_4dirs[local_idx] = local_max_4dirs[index(width, height, bx * 8 + threadIdx.x, by * 8 + threadIdx.y)];

		__syncthreads();

		// Active threads update their cones
		if (!finished) {
			for (int i = 0; i < 8; ++i) {
				for (int j = 0; j < 8; ++j) {
					int du = bx * 8 + j - u;
					int dv = by * 8 + i - v;

					// TODO the first block separately, then only the side the block is on
					int discrete_dir = abs(du) >= abs(dv) ? (du >= 0 ? 0 : 2) : (dv >= 0 ? 1 : 3);

					if (!(l_local_max_4dirs[i * 8 + j] != discrete_dir)) continue;

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
			if (r * r * 64 > (1.0f - h) * (1.0f - h) * ratio2 * max(width, height) * max(width, height))
				finished = true;

			// warp-level check
			unsigned mask = __ballot_sync(0xffffffff, finished);
			bool warp_finished = (mask == 0xffffffff);

			// communicate between warps
			if (lane_idx == 0 && warp_finished) atomicOr(&block_finished_flags, 1u << warp_idx);

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

	unsigned char dhdu = (hsum + 1020) / 8;
	unsigned char dhdv = (vsum + 1020) / 8;

	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<unsigned char>(ratio * 255);
	cone_map[idx * 4 + 2] = dhdu;
	cone_map[idx * 4 + 3] = dhdv;
}
