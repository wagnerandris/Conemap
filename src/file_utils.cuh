#pragma once

// STD
#include <stdio.h>

// STB
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "external/stb_image.h"

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "external/stb_image_write.h"

// CUDA
#include <cuda_runtime.h>

// CUDA error check macro
#ifndef CUDA_CHECK
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(code);
	}
}
#endif

#include "Texture.cuh"

inline TextureDevicePointer<unsigned char> read_texture_to_device(const char* filepath) {
	int width;
	int height;
	int channels;
	unsigned char* data;
	// load data
	data = stbi_load(filepath, &width, &height, &channels, 1);
	if (!data) {
		fprintf(stderr, "Could not load texture from %s.\n", filepath);
		return TextureDevicePointer<unsigned char>{0, 0, 0, nullptr};
	}

	printf("Loaded texture from %s.\nWidth: %d, Height: %d\n", filepath, width, height);
	return TextureDevicePointer<unsigned char>{width, height, 1, data};
	free(data);
}

inline bool write_device_texture_to_file(const char* filepath, TextureDevicePointer<unsigned char> &tex) {
	// copy to host
	int size = tex.width * tex.height * tex.channels;
	unsigned char* h_data = new unsigned char[size];
	CUDA_CHECK(cudaMemcpy(h_data, *tex, size, cudaMemcpyDeviceToHost));

	// write to file
	if (!stbi_write_png(filepath, tex.width, tex.height, tex.channels, h_data, tex.width * tex.channels)) {
		fprintf(stderr, "Could not write image to %s\n", filepath);
		delete[] h_data;
		return false;
	};
	delete[] h_data;
	printf("Written texture as %s\n", filepath);
	return true;
}
