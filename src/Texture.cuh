#pragma once

// STD
#include <stdio.h>

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

template<typename T>
struct TextureView {
	const int width;
	const int height;
	const int channels;

__device__ inline				T& operator[](std::size_t idx)			 { return device_pointer[idx]; }
__device__ inline const T& operator[](std::size_t idx) const { return device_pointer[idx]; }

// private:
	T* device_pointer;
};

template<typename T>
struct Texture {
	const int width;
	const int height;
	const int channels;

	Texture(int width_, int height_, int channels_): width(width_), height(height_), channels(channels_) {
		size_t size = width * height * channels * sizeof(T);
		CUDA_CHECK(cudaMalloc(&device_pointer, size));
	}

	Texture(int width_, int height_, int channels_, T* data): width(width_), height(height_), channels(channels) {
		size_t size = width * height * channels * sizeof(T);
		CUDA_CHECK(cudaMalloc(&device_pointer, size));
		CUDA_CHECK(cudaMemcpy(device_pointer, data, size, cudaMemcpyHostToDevice));
	}

	~Texture() {
		CUDA_CHECK(cudaFree(device_pointer));
	}

	// Too cursed?
	TextureView<T> operator()() {
		return {width, height, channels, device_pointer};
	}

	TextureView<T> view() {
		return {width, height, channels, device_pointer};
	}

// private:
	T* device_pointer;
};
