#pragma once

// STD
#include <stdio.h>

// STB
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

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

// TODO more sophisticated checks? (file exists etc)
inline bool read_texture_to_device(unsigned char* &device_pointer, const char* filepath, int* width, int* height, int* channels) {
	// load data
  unsigned char* data = stbi_load(filepath, width, height, channels, 1);
  if (!data) {
    fprintf(stderr, "Could not load texture from %s.\n", filepath);
    return false;
  }

	// copy to device
	int size = (*width) * (*height);
  CUDA_CHECK(cudaMalloc(&device_pointer, size));
  CUDA_CHECK(cudaMemcpy(device_pointer, data, size, cudaMemcpyHostToDevice));
  free(data);

  printf("Loaded texture from %s.\nWidth: %d, Height: %d\n", filepath, *width, *height);
  return true;
}

inline bool write_device_texture_to_file(const char* filepath, unsigned char* device_pointer, int width, int height, int channels) {
	// copy to host
	int size = width * height * channels;
  unsigned char* h_data = new unsigned char[size];
  CUDA_CHECK(cudaMemcpy(h_data, device_pointer, size, cudaMemcpyDeviceToHost));

	// write to file
  if (stbi_write_png(filepath, width, height, channels, h_data, width * channels)) {
		printf("Written image as %s\n", filepath);
		delete[] h_data;
		return true;
  };
  return false;
}
