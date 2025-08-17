// STB
#include <cmath>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

// STD
#include <stdio.h>
#include <filesystem>
#include <string>

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

__global__ void first_derivative(unsigned char* input, unsigned char* output_image, int* fod, int width, int height, int channels) {
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
			hsum += input[cidx] * hkernel[dv * 3 + du];
			vsum += input[cidx] * vkernel[dv * 3 + du];
		}
	}

	// int grad[2] = {hsum, vsum};
	// int scale = std::sqrt(vsum * vsum + hsum * hsum);
	// float dir = std::atan2(vsum, hsum);

	int idx = (v * width + u) * 4;
	output_image[idx + 0] = (hsum + 1020) >> 3;
	output_image[idx + 1] = (vsum + 1020) >> 3;
	output_image[idx + 2] = 127;
	output_image[idx + 3] = 255;

	idx = (v * width + u) * 2;
	fod[idx + 0] = hsum;
	fod[idx + 1] = vsum;
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
	
	int val = - (hhsum * fod[idx * 2 + 1] * fod[idx * 2 + 1] - (hvsum + vhsum) * fod[idx * 2 + 0] * fod[idx * 2 + 1] + vvsum * fod[idx * 2 + 0] * fod[idx * 2 + 0]);

	// TODO normalize
	watershed_image[idx * 4 + 0] = val;
	watershed_image[idx * 4 + 1] = val;
	watershed_image[idx * 4 + 2] = val;
	watershed_image[idx * 4 + 3] = 255;
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
  size_t input_size = width * height * channels;
  size_t output_size = width * height * 4;
  unsigned char *d_input, *d_derivative_image;
  int* d_fod;

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&d_input, input_size));
  CUDA_CHECK(cudaMalloc(&d_derivative_image, output_size));
  CUDA_CHECK(cudaMalloc(&d_fod, width * height * 2 * sizeof(int)));

  // Copy image to GPU
  CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));

  // Launch kernel
  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
  first_derivative<<<blocks, threads>>>(d_input, d_derivative_image, d_fod, width, height, channels);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back
  unsigned char* h_derivative_image = (unsigned char*)malloc(output_size);
  CUDA_CHECK(cudaMemcpy(h_derivative_image, d_derivative_image, output_size, cudaMemcpyDeviceToHost));

  // Save image
	std::string derivative_file = std::filesystem::path(filepath).stem().string() + "_derivative.png";
  stbi_write_png(derivative_file.c_str(), width, height, 4, h_derivative_image, width * 4);

  printf("Written image as %s\n", derivative_file.c_str());

// Second order derivatives and watershed
	unsigned char *d_second_derivative_image, *d_watershed;

  CUDA_CHECK(cudaMalloc(&d_second_derivative_image, output_size));
  CUDA_CHECK(cudaMalloc(&d_watershed, output_size));
  
  // Launch kernel
  second_derivative<<<blocks, threads>>>(d_fod, d_second_derivative_image, d_watershed, width, height);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result back
  unsigned char* h_second_derivative_image = (unsigned char*)malloc(output_size);
  CUDA_CHECK(cudaMemcpy(h_second_derivative_image, d_second_derivative_image, output_size, cudaMemcpyDeviceToHost));
  unsigned char* h_watershed = (unsigned char*)malloc(output_size);
  CUDA_CHECK(cudaMemcpy(h_watershed, d_watershed, output_size, cudaMemcpyDeviceToHost));

  // Save image
	std::string second_derivative_file = std::filesystem::path(filepath).stem().string() + "_second_derivative.png";
  stbi_write_png(second_derivative_file.c_str(), width, height, 4, h_second_derivative_image, width * 4);
  printf("Written image as %s\n", second_derivative_file.c_str());
	
	std::string watershed_file = std::filesystem::path(filepath).stem().string() + "_watershed.png";
  stbi_write_png(watershed_file.c_str(), width, height, 4, h_watershed, width * 4);
  printf("Written image as %s\n", watershed_file.c_str());

  // Cleanup
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_derivative_image));
  CUDA_CHECK(cudaFree(d_fod));
  CUDA_CHECK(cudaFree(d_second_derivative_image));
  CUDA_CHECK(cudaFree(d_watershed));
  stbi_image_free(h_input);
  free(h_derivative_image);
  free(h_second_derivative_image);
  free(h_watershed);
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
