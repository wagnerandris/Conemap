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
struct TextureDevicePointer {
	const int width;
	const int height;
	const int channels;

	TextureDevicePointer(int width_, int height_, int channels_): width(width_), height(height_), channels(channels_) {
		size_t size = width * height * channels * sizeof(T);
		CUDA_CHECK(cudaMalloc(&device_pointer, size));
	}

	TextureDevicePointer(int width_, int height_, int channels_, T* data): width(width_), height(height_), channels(channels_) {
		size_t size = width * height * channels * sizeof(T);
		CUDA_CHECK(cudaMalloc(&device_pointer, size));
		CUDA_CHECK(cudaMemcpy(device_pointer, data, size, cudaMemcpyHostToDevice));
	}
	
	TextureDevicePointer(TextureDevicePointer&& other) noexcept // move constructor
		: width(other.width), height(other.height), channels(other.channels), device_pointer(std::exchange(other.device_pointer, nullptr)) {}

	TextureDevicePointer& operator=(TextureDevicePointer&& other) noexcept = delete;
	TextureDevicePointer(const TextureDevicePointer& other) = delete;
	TextureDevicePointer& operator=(const TextureDevicePointer& other) = delete;

	~TextureDevicePointer() {
		CUDA_CHECK(cudaFree(device_pointer));
	}

	operator bool() {return device_pointer;}

	T* operator*() {return device_pointer;}

private:
	T* device_pointer;
};


template<typename T>
struct TextureHostPointer {
	const int width;
	const int height;
	const int channels;

	TextureHostPointer(int width_, int height_, int channels_): width(width_), height(height_), channels(channels_) {
		size_t size = width * height * channels;
		host_pointer = new T[size];
		memset(host_pointer, 0, size * sizeof(T));
	}
	
	TextureHostPointer(int width_, int height_, int channels_, T* data_): width(width_), height(height_), channels(channels_), host_pointer(data_) {}

	TextureHostPointer(TextureDevicePointer<T> &tdp): width(tdp.width), height(tdp.height), channels(tdp.channels) {
		size_t size = width * height * channels;
		host_pointer = new T[size];
		CUDA_CHECK(cudaMemcpy(host_pointer, *tdp, size * sizeof(T), cudaMemcpyDeviceToHost));
	}
	
	TextureHostPointer(TextureHostPointer&& other) noexcept // move constructor
		: width(other.width), height(other.height), channels(other.channels), host_pointer(std::exchange(other.host_pointer, nullptr)) {}

	TextureHostPointer& operator=(TextureHostPointer&& other) noexcept = delete;
	TextureHostPointer(const TextureHostPointer& other) = delete;
	TextureHostPointer& operator=(const TextureHostPointer& other) = delete;

	~TextureHostPointer() {
		delete[] host_pointer;
	}

	operator bool() {return host_pointer;}

	T* operator*() {return host_pointer;}

private:
	T* host_pointer;
};
