// STB
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

// STD
#include <stdio.h>
#include <filesystem>
#include <string>
#include <cmath>
#include <cstdlib>

// Boost
#include <boost/program_options.hpp>

// CUDA
#include <cuda_runtime.h>

// CUDA error check macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

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

__global__ void local_max_dirs(unsigned char* heightmap, unsigned char* dirs, int width, int height, int channels) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;

  if (u >= width || v >= height) return;

	int h = heightmap[(v * width + u) * channels];

    // Direction vectors: Right, Top-Right, Top, Top-Left, Left, Bottom-Left, Bottom, Bottom-Right
    int du[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
    int dv[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

    unsigned char result = 0;

    for (int dir = 0; dir < 8; ++dir) {
        int un = u + du[dir];
        int up = u - du[dir];
        int vn = v + dv[dir];
        int vp = v - dv[dir];

        // Ensure both neighbors are within bounds
        bool u_good = (un >= 0 && un < width && vn >= 0 && vn < height);
        bool v_good = (up >= 0 && up < width && vp >= 0 && vp < height);

				// edges are always max
				// TODO looped textures?
        int hn = u_good ? heightmap[(vn * width + un) * channels] : -1;
        int hp = v_good ? heightmap[(vp * width + up) * channels] : -1;

				// local max in hp to hn direction (if there is a plateau, we need it's last point)
        if (h > hn && h >= hp) {
            result |= (1 << dir);
        }
    }

    dirs[v * width + u] = result;
}

__global__ void create_cone_map2(unsigned char* heightmap, unsigned char* derivative_image, unsigned char* dirs, unsigned char* cone_map, int width, int height, int channels) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;

  if (u >= width || v >= height) return;

	float iwidth = 1.0f / width;
	float iheight = 1.0f / height;

	// TODO Why are we assuming run/rise = 1 exactly (instead of infinity)? Float textures?
	float min_ratio2 = 1.0f;

	// normalize height
	float h = heightmap[(v * width + u) * channels] / 255.0f;

	// init variables
	int du, dv;
	float dun, dvn;
	int start, end;

	// search in an increasing radius spiral around the texel
	for (int rad = 1;
			// TODO why the 1.1f?
			// otherwise we can't find anything steeper than the current min_ratio (see min_ratio assignment)
			rad * rad <= 1.1f * (1.0f - h) * width *
									 1.1f * (1.0f - h) * height *
									 min_ratio2 &&
			// because we started from 1, and further than (1.0f - h) * width, we couldn't find anything steeper than 1
			rad <= 1.1f * (1.0f - h) * width &&
			rad <= 1.1f * (1.0f - h) * height;
			++rad) {

		// Right side

		// u displacement
		du = u + rad;
		// normalized
		dun = rad * iwidth;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (du >= width) du -= width; 
		// set v limits
		start = max(v - rad, 0);
		end = min(v + rad - 1, height);

		// go through side
		for (int dv = start; dv < end; ++dv) {
			// check if local maxima in the given direction, skip if not

			int discrete_dir;
			if (dv - start <= rad / 2) discrete_dir = 1;
			else if (end - dv <= rad / 2) discrete_dir = 7;
			else discrete_dir = 0;

			// float dir = atan2f(dv, du);
			// unsigned char discrete_dir = static_cast<unsigned char>(
			// 													(dir + M_PI // all positive
			// 													+ M_PI_4f / 2.0f) // align regions
			// 													/ M_PI_4f); // 8 dirs
			// if (dirs[(dv * width + du)] & 1 << discrete_dir) continue;

			// normalize v displacement
			dvn = (dv - v) * iheight;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}

		// Top side

		// u displacement
		dv = v - rad;
		// normalized
		dvn = -rad * iheight;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (dv < 0) dv += height; 
		// set u limits
		start = max(u - rad, 0);
		end = min(u + rad - 1, width);

		// go through side
		for (int du = start; du < end; ++du) {
			// check if local maxima in the given direction, skip if not

			int discrete_dir;
			if (du - start <= rad / 2) discrete_dir = 3;
			else if (end - du <= rad / 2) discrete_dir = 1;
			else discrete_dir = 2;

			// float dir = atan2f(dv, du);
			// unsigned char discrete_dir = static_cast<unsigned char>(
			// 													(dir + M_PI // all positive
			// 													+ M_PI_4f / 2.0f) // align regions
			// 													/ M_PI_4f); // 8 dirs
			// if (dirs[(dv * width + du)] & 1 << discrete_dir) continue;

			// normalize v displacement
			dun = (du - u) * iwidth;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}

		// Left side

		// u displacement
		du = u - rad;
		// normalized
		dun = -rad * iwidth;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (du < 0) du += width; 
		// set v limits
		start = max(v - rad + 1, 0);
		end = min(v + rad, height);

		// go through side
		for (int dv = start; dv < end; ++dv) {
			// check if local maxima in the given direction, skip if not

			int discrete_dir;
			if (dv - start <= rad / 2) discrete_dir = 3;
			else if (end - dv <= rad / 2) discrete_dir = 5;
			else discrete_dir = 4;

			// float dir = atan2f(dv, du);
			// unsigned char discrete_dir = static_cast<unsigned char>(
			// 													(dir + M_PI // all positive
			// 													+ M_PI_4f / 2.0f) // align regions
			// 													/ M_PI_4f); // 8 dirs
			// if (dirs[(dv * width + du)] & 1 << discrete_dir) continue;

			// normalize v displacement
			dvn = (dv - v) * iheight;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}


		// Bottom side

		// u displacement
		dv = v + rad;
		// normalized
		dvn = rad * iheight;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (dv >= height) dv -= height; 
		// set u limits
		start = max(u - rad + 1, 0);
		end = min(u + rad, width);

		// go through side
		for (int du = start; du < end; ++du) {
			// check if local maxima in the given direction, skip if not

			int discrete_dir;
			if (du - start <= rad / 2) discrete_dir = 5;
			else if (end - du <= rad / 2) discrete_dir = 7;
			else discrete_dir = 6;

			// float dir = atan2f(dv, du);
			// unsigned char discrete_dir = static_cast<unsigned char>(
			// 													(dir + M_PI // all positive
			// 													+ M_PI_4f / 2.0f) // align regions
			// 													/ M_PI_4f); // 8 dirs
			// if (dirs[(dv * width + du)] & 1 << discrete_dir) continue;

			// normalize v displacement
			dun = (du - u) * iwidth;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}
	}

	float ratio = sqrt(min_ratio2);
  // most of the data is on the low end...sqrting again spreads it better
  // (plus multiply is a cheap operation in shaders!)
  // -- Dummer
	ratio = sqrt(ratio);
	cone_map[(v * width + u) * 4 + 0] = heightmap[(v * width + u) * channels];
	cone_map[(v * width + u) * 4 + 1] = static_cast<unsigned char>(ratio * 255);
	cone_map[(v * width + u) * 4 + 2] = derivative_image[(v * width + u) * 3];
	cone_map[(v * width + u) * 4 + 3] = derivative_image[(v * width + u) * 3 + 1];
}

__global__ void first_derivative(unsigned char* heightmap, unsigned char* derivative_image, unsigned char* dirs, unsigned char* dirs_image, int* fod, int width, int height, int channels) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;

  if (u >= width || v >= height) return;
	
  // Possible optimizations:
  //  - For every pixel, compute the sum with the side adjacent ones in both dimensions and save to shared memory
	//  - Step by channels instead of 1 in the loop (only worth it, if kernel lookup is independent)
	//  - Previtt operator instead of Sobel -> no kernel needed multiply by value/sign of index
	int hsum = 0;
	int vsum = 0;
	for (int dv = 0; dv < 3; ++dv) {
		for (int du = 0; du < 3; ++du) {
      int cu = min(max(u + du - 1, 0), width - 1);
      int cv = min(max(v + dv - 1, 0), height - 1);
      int cidx = (cv * width + cu) * channels;
			hsum += heightmap[cidx] * hkernel[dv * 3 + du];
			vsum += heightmap[cidx] * vkernel[dv * 3 + du];
		}
	}

	// int grad[2] = {hsum, vsum};
	// int scale = std::sqrt(vsum * vsum + hsum * hsum);
	float dir = atan2f(vsum, hsum);
	unsigned char discrete_dir = static_cast<unsigned char>(
																(dir + M_PI // all positive
																+ M_PI_4f / 2.0f) // align regions
																/ M_PI_4f // 8 dirs
																) % 4; // the 4 we care about;

	int idx = v * width + u;

	derivative_image[idx * 3 + 0] = (hsum + 1020) >> 3;
	derivative_image[idx * 3 + 1] = (vsum + 1020) >> 3;
	derivative_image[idx * 3 + 2] = 127;

	fod[idx * 2 + 0] = hsum;
	fod[idx * 2 + 1] = vsum;

	dirs[v * width + u] = discrete_dir;

	dirs_image[v * width + u] = dirs[v * width + u] * 32 + 127;
}

__global__ void second_derivative(int* fod, unsigned char* sod_image, unsigned char* watershed_image, int width, int height) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;

  if (u >= width || v >= height) return;
	
  // Possible optimizations:
  //  - For every pixel, compute the sum with the side adjacent ones in both dimensions and save to shared memory
	//  - Step by channels instead of 1 in the loop (only worth it, if kernel lookup is independent)
	//  - Previtt operator instead of Sobel -> no kernel needed multiply by value/sign of index
	int hhsum = 0;
	int vhsum = 0;
	int hvsum = 0;
	int vvsum = 0;
	for (int dv = 0; dv < 3; ++dv) {
		for (int du = 0; du < 3; ++du) {
      int cu = min(max(u + du - 1, 0), width - 1);
      int cv = min(max(v + dv - 1, 0), height - 1);
      int cidx = (cv * width + cu) * 2;
			hhsum += fod[cidx + 0] * hkernel[dv * 3 + du];
			hvsum += fod[cidx + 1] * hkernel[dv * 3 + du];
			vhsum += fod[cidx + 0] * vkernel[dv * 3 + du];
			vvsum += fod[cidx + 1] * vkernel[dv * 3 + du];
		}
	}

	int idx = v * width + u;

	sod_image[idx * 4 + 0] = (hhsum + 1020) >> 3;
	sod_image[idx * 4 + 1] = (hvsum + 1020) >> 3;
	sod_image[idx * 4 + 2] = (vhsum + 1020) >> 3;
	sod_image[idx * 4 + 3] = (vvsum + 1020) >> 3;
	
	int val = hhsum * fod[idx * 2 + 1] * fod[idx * 2 + 1] - (hvsum + vhsum) * fod[idx * 2 + 0] * fod[idx * 2 + 1] + vvsum * fod[idx * 2 + 0] * fod[idx * 2 + 0];

	watershed_image[idx] = val < 0 ? 255 : 0;
}

__global__ void non_maximum_suppression(unsigned char* heightmap, unsigned char* dirs, unsigned char* watershed_image, unsigned char* suppressed_image, int width, int height, int channels) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;

  if (u >= width || v >= height) return;

	int du;
	int dv;

	switch (dirs[v * width + u]) {
		case 0:
			du = 0;
			dv = 1;
			break;
		case 1:
			du = -1;
			dv = 1;
			break;
		case 2:
			du = 1;
			dv = 0;
			break;
		case 3:
			du = 1;
			dv = 1;
			break;
	}

	if (watershed_image[v * width + u] == 0) {
		suppressed_image[v * width + u] = 0;
	} else
	if (u + du >= 0 && u + du < width &&
			v + dv >= 0 && v + dv < height &&
			heightmap[((v + dv) * width + u + du) * channels] > heightmap[(v * width + u) * channels]) {
			suppressed_image[v * width + u] = 0;
	} else
	if (u - du >= 0 && u - du < width &&
			v - dv >= 0 && v - dv < height &&
			heightmap[((v - dv) * width + u - du) * channels] > heightmap[(v * width + u) * channels]) {
			suppressed_image[v * width + u] = 0;
	} else {
		suppressed_image[v * width + u] = 255;
	}
}

__global__ void create_cone_map(unsigned char* heightmap, unsigned char* derivative_image, unsigned char* suppressed_image, unsigned char* cone_map, int width, int height, int channels) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;

  if (u >= width || v >= height) return;

	float iwidth = 1.0f / width;
	float iheight = 1.0f / height;

	// TODO Why are we assuming run/rise = 1, instead of infinity? Float textures?
	float min_ratio2 = 1.0f;

	// normalize height
	float h = heightmap[(v * width + u) * channels] / 255.0f;

	// init variables
	int du, dv;
	float dun, dvn;
	int start, end;

	// search in an increasing radius spiral around the texel
	for (int rad = 1;
			// TODO why the 1.1f?
			// otherwise we can't find anything steeper than the current min_ratio (see min_ratio assignment)
			rad * rad <= 1.1f * (1.0f - h) * width *
									 1.1f * (1.0f - h) * height *
									 min_ratio2 &&
			// because we started from 1, and further than (1.0f - h) * width, we couldn't find anything steeper than 1
			rad <= 1.1f * (1.0f - h) * width &&
			rad <= 1.1f * (1.0f - h) * height;
			++rad) {

		// Right side

		// u displacement
		du = u + rad;
		// normalized
		dun = rad * iwidth;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (du >= width) du -= width; 
		// set v limits
		start = max(v - rad, 0);
		end = min(v + rad - 1, height);

		// go through side
		for (int dv = start; dv < end; ++dv) {
			// check if (suppressed) watershed point, skip if not
			if (suppressed_image[(dv * width + du)] == 0) continue;

			// normalize v displacement
			dvn = (dv - v) * iheight;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}

		// Top side

		// u displacement
		dv = v - rad;
		// normalized
		dvn = -rad * iheight;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (dv < 0) dv += height; 
		// set u limits
		start = max(u - rad, 0);
		end = min(u + rad - 1, width);

		// go through side
		for (int du = start; du < end; ++du) {
			// check if (suppressed) watershed point, skip if not
			if (suppressed_image[(dv * width + du)] == 0) continue;

			// normalize v displacement
			dun = (du - u) * iwidth;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}

		// Left side

		// u displacement
		du = u - rad;
		// normalized
		dun = -rad * iwidth;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (du < 0) du += width; 
		// set v limits
		start = max(v - rad + 1, 0);
		end = min(v + rad, height);

		// go through side
		for (int dv = start; dv < end; ++dv) {
			// check if (suppressed) watershed point, skip if not
			if (suppressed_image[(dv * width + du)] == 0) continue;

			// normalize v displacement
			dvn = (dv - v) * iheight;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}


		// Bottom side

		// u displacement
		dv = v + rad;
		// normalized
		dvn = rad * iheight;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (dv >= height) dv -= height; 
		// set u limits
		start = max(u - rad + 1, 0);
		end = min(u + rad, width);

		// go through side
		for (int du = start; du < end; ++du) {
			// check if (suppressed) watershed point, skip if not
			if (suppressed_image[(dv * width + du)] == 0) continue;

			// normalize v displacement
			dun = (du - u) * iwidth;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}
	}

	float ratio = sqrt(min_ratio2);
  // most of the data is on the low end...sqrting again spreads it better
  // (plus multiply is a cheap operation in shaders!)
  // -- Dummer
	ratio = sqrt(ratio);
	cone_map[(v * width + u) * 4 + 0] = heightmap[(v * width + u) * channels];
	cone_map[(v * width + u) * 4 + 1] = static_cast<unsigned char>(ratio * 255);
	cone_map[(v * width + u) * 4 + 2] = derivative_image[(v * width + u) * 3];
	cone_map[(v * width + u) * 4 + 3] = derivative_image[(v * width + u) * 3 + 1];
}

unsigned char* read_texture_from_file(const char* filepath, int* w, int* h, int* c) {
  unsigned char* data = stbi_load(filepath, w, h, c, 0);
  if (!data) {
    fprintf(stderr, "Could not load texture from %s.\n", filepath);
    return nullptr;
  }
  printf("Loaded texture from %s.\nWidth: %d, Height: %d, Channels: %d\n", filepath, *w, *h, *c);
  return data;
}

void convert_image(const char* filepath) {
  int width, height, channels;

// Load image
  unsigned char* h_input = read_texture_from_file(filepath, &width, &height, &channels);
  if (!h_input) return;

// First order derivatives
  size_t size = width * height;
  size_t input_size = size * channels;
  unsigned char *d_input, *d_derivative_image, *d_derivative_dirs, *d_derivative_dirs_image;
  int* d_fod;

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&d_input, input_size));
  CUDA_CHECK(cudaMalloc(&d_derivative_image, size * 3));
  CUDA_CHECK(cudaMalloc(&d_derivative_dirs, size));
  CUDA_CHECK(cudaMalloc(&d_derivative_dirs_image, size));
  CUDA_CHECK(cudaMalloc(&d_fod, size * 2 * sizeof(int)));

  // Copy image to GPU
  CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));

  // Launch kernel
  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  first_derivative<<<blocks, threads>>>(d_input, d_derivative_image, d_derivative_dirs, d_derivative_dirs_image, d_fod, width, height, channels);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back
  unsigned char* h_derivative_image = (unsigned char*)malloc(size * 3);
  CUDA_CHECK(cudaMemcpy(h_derivative_image, d_derivative_image, size * 3, cudaMemcpyDeviceToHost));
  unsigned char* h_dirs_image = (unsigned char*)malloc(size);
  CUDA_CHECK(cudaMemcpy(h_dirs_image, d_derivative_dirs_image, size, cudaMemcpyDeviceToHost));

  // Save image
	std::string derivative_file = std::filesystem::path(filepath).stem().string() + "_derivative.png";
  stbi_write_png(derivative_file.c_str(), width, height, 3, h_derivative_image, width * 3);
	printf("Written image as %s\n", derivative_file.c_str());
	
	std::string dirs_image_file = std::filesystem::path(filepath).stem().string() + "_derivative_dirs_image.png";
  stbi_write_png(dirs_image_file.c_str(), width, height, 1, h_dirs_image, width * 1);
  printf("Written image as %s\n", dirs_image_file.c_str());

// // Second order derivatives and watershed
// 	unsigned char *d_second_derivative_image, *d_watershed;
//
//   CUDA_CHECK(cudaMalloc(&d_second_derivative_image, size * 4));
//   CUDA_CHECK(cudaMalloc(&d_watershed, size));
//
//   // Launch kernel
//   second_derivative<<<blocks, threads>>>(d_fod, d_second_derivative_image, d_watershed, width, height);
//   CUDA_CHECK(cudaDeviceSynchronize());
//
//   // Copy result back
//   unsigned char* h_second_derivative_image = (unsigned char*)malloc(size * 4);
//   CUDA_CHECK(cudaMemcpy(h_second_derivative_image, d_second_derivative_image, size * 4, cudaMemcpyDeviceToHost));
//   unsigned char* h_watershed = (unsigned char*)malloc(size);
//   CUDA_CHECK(cudaMemcpy(h_watershed, d_watershed, size, cudaMemcpyDeviceToHost));
//
//   // Save image
// 	std::string second_derivative_file = std::filesystem::path(filepath).stem().string() + "_second_derivative.png";
//   stbi_write_png(second_derivative_file.c_str(), width, height, 4, h_second_derivative_image, width * 4);
//   printf("Written image as %s\n", second_derivative_file.c_str());
//
// 	std::string watershed_file = std::filesystem::path(filepath).stem().string() + "_watershed.png";
//   stbi_write_png(watershed_file.c_str(), width, height, 1, h_watershed, width * 1);
//   printf("Written image as %s\n", watershed_file.c_str());
//
// // Non maximum suppression
// 	unsigned char *d_suppressed;
//
//   CUDA_CHECK(cudaMalloc(&d_suppressed, size));
//
//   // Launch kernel
//   non_maximum_suppression<<<blocks, threads>>>(d_input, d_dirs, d_watershed, d_suppressed, width, height, channels);
//   CUDA_CHECK(cudaDeviceSynchronize());
//
//   // Copy result back
//   unsigned char* h_suppressed = (unsigned char*)malloc(size);
//   CUDA_CHECK(cudaMemcpy(h_suppressed, d_suppressed, size, cudaMemcpyDeviceToHost));
//
//   // Save image
// 	std::string suppressed_file = std::filesystem::path(filepath).stem().string() + "_suppressed.png";
//   stbi_write_png(suppressed_file.c_str(), width, height, 1, h_suppressed, width * 1);
//   printf("Written image as %s\n", suppressed_file.c_str());
//
// // Relaxed cone map generation
// 	unsigned char *d_cone_map;
//
//   CUDA_CHECK(cudaMalloc(&d_cone_map, size * 4));
//
//   // Launch kernel
//   create_cone_map<<<blocks, threads>>>(d_input, d_derivative_image, d_suppressed, d_cone_map, width, height, channels);
//   CUDA_CHECK(cudaDeviceSynchronize());
//
//   // Copy result back
//   unsigned char* h_cone_map = (unsigned char*)malloc(size * 4);
//   CUDA_CHECK(cudaMemcpy(h_cone_map, d_cone_map, size * 4, cudaMemcpyDeviceToHost));
//
//   // Save image
// 	std::string cone_map_file = std::filesystem::path(filepath).stem().string() + "_relaxed_cone_map.png";
//   stbi_write_png(cone_map_file.c_str(), width, height, 4, h_cone_map, width * 4);
//   printf("Written image as %s\n", cone_map_file.c_str());

// Directional local maxima
  // Allocate device memory
	unsigned char *d_local_max_dirs;
  CUDA_CHECK(cudaMalloc(&d_local_max_dirs, size));
  
  // Launch kernel
  local_max_dirs<<<blocks, threads>>>(d_input, d_local_max_dirs, width, height, channels);
  CUDA_CHECK(cudaDeviceSynchronize());	
	

// Relaxed cone map generation
	unsigned char *d_cone_map;

  CUDA_CHECK(cudaMalloc(&d_cone_map, size * 4));
  
  // Launch kernel
  create_cone_map2<<<blocks, threads>>>(d_input, d_derivative_image, d_local_max_dirs, d_cone_map, width, height, channels);
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Copy result back
  unsigned char* h_cone_map = (unsigned char*)malloc(size * 4);
  CUDA_CHECK(cudaMemcpy(h_cone_map, d_cone_map, size * 4, cudaMemcpyDeviceToHost));

  // Save image
	std::string cone_map_file = std::filesystem::path(filepath).stem().string() + "_relaxed_cone_map.png";
  stbi_write_png(cone_map_file.c_str(), width, height, 4, h_cone_map, width * 4);
  printf("Written image as %s\n", cone_map_file.c_str());


// Cleanup
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_derivative_image));
  CUDA_CHECK(cudaFree(d_derivative_dirs));
  CUDA_CHECK(cudaFree(d_derivative_dirs_image));
  CUDA_CHECK(cudaFree(d_fod));
  // CUDA_CHECK(cudaFree(d_second_derivative_image));
  // CUDA_CHECK(cudaFree(d_watershed));
  // CUDA_CHECK(cudaFree(d_suppressed));
  CUDA_CHECK(cudaFree(d_local_max_dirs));
  CUDA_CHECK(cudaFree(d_cone_map));
  stbi_image_free(h_input);
  free(h_derivative_image);
  free(h_dirs_image);
  // free(h_second_derivative_image);
  // free(h_watershed);
  // free(h_suppressed);
  free(h_cone_map);
}

int main(int argc, char** argv) {
	if (argc < 2) {
		fprintf(stderr, "No texture provided.\n");
		exit(0);
	}
	for (int i = 1; i < argc; ++i) {
		if (!std::filesystem::exists(argv[i]) && std::filesystem::is_regular_file(argv[i])) {
			fprintf(stderr,"No such file: %s\n", argv[i]);
		}
		convert_image(argv[i]);
	}

  return 0;
}
