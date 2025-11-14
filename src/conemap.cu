#include "conemap.hpp"

#include "file_utils.cuh"
#include "kernels.cuh"
#include "Texture.cuh"


void conemap::analytic(std::filesystem::path output_path, std::string filepath, bool depthmap) {

	std::string output_name = depthmap ?
		output_path / std::filesystem::path(filepath).stem().concat("_depthmap") :
		output_path / std::filesystem::path(filepath).stem();

/* Load image */
	TextureDevicePointer<unsigned char> input_image = read_texture_to_device(filepath.c_str());
	if (!input_image) return;

	int width = input_image.width;
	int height = input_image.height;

	// Threads/blocks
	// TODO what's optimal?
	dim3 threads(16, 16);
	dim3 blocks((width + threads.x - 1) / threads.x,
							(height + threads.y - 1) / threads.y);

	if (depthmap) {
		invert<<<blocks, threads>>>(*input_image, width, height);
	}

/* First order derivatives */
	// Allocate device memory
	TextureDevicePointer<int> fods{width, height, 2};
	TextureDevicePointer<float> fod_dirs{width, height, 1};
	
	// Launch kernel
	fod<<<blocks, threads>>>(*input_image, *fods, *fod_dirs, width, height);
	CUDA_CHECK(cudaDeviceSynchronize());

/* Second order derivatives and watershed */
	// Allocate device memory
	TextureDevicePointer<bool> watersheds{width, height, 1};

	// Launch kernel
	watershed<<<blocks, threads>>>(*fods, *watersheds, width, height);
	CUDA_CHECK(cudaDeviceSynchronize());

/* Non maximum suppression */
	// Allocate device memory
	TextureDevicePointer<bool> suppressed{width, height, 1};

	// Launch kernel
	non_maximum_suppression<<<blocks, threads>>>(*input_image, *fod_dirs, *watersheds, *suppressed, width, height);
	CUDA_CHECK(cudaDeviceSynchronize());

/* Relaxed cone map generation: Baseline */
	// Allocate device memory
	TextureDevicePointer<unsigned char> cone_map{width, height, 4};

	// Launch kernel
	create_cone_map_analytic<<<blocks, threads>>>(*input_image, *suppressed, *fod_dirs, *fods, *cone_map, width, height);
	CUDA_CHECK(cudaDeviceSynchronize());
	
/* Write result image to file */
	write_device_texture_to_file((output_name + "_relaxed_cone_map_analytic.png").c_str(), cone_map);
}


void conemap::discrete(std::filesystem::path output_path, std::filesystem::path filepath, bool depthmap) {

	std::string output_name = depthmap ?
		output_path / filepath.stem().concat("_depthmap") :
		output_path / filepath.stem();

/* Load image */
	TextureDevicePointer<unsigned char> input_image = read_texture_to_device(filepath.c_str());
	if (!input_image) return;

	int width = input_image.width;
	int height = input_image.height;

	// Threads/blocks
	// TODO what's optimal?
	dim3 threads(16, 16);
	dim3 blocks((width + threads.x - 1) / threads.x,
							(height + threads.y - 1) / threads.y);

	if (depthmap) {
		invert<<<blocks, threads>>>(*input_image, width, height);
	}

/* Directional local maxima */
	// Allocate device memory
	TextureDevicePointer<unsigned char> local_max_8dirs{width, height, 1};

	// Launch kernel
	local_max_8dir<<<blocks, threads>>>(*input_image, *local_max_8dirs, width, height);
	CUDA_CHECK(cudaDeviceSynchronize());

/* Relaxed cone map generation: Discrete directions */
	// Allocate device memory
	TextureDevicePointer<unsigned char> cone_map{width, height, 4};

	// Launch kernel
	create_cone_map_8dir<<<blocks, threads>>>(*input_image, *local_max_8dirs, *cone_map, width, height);
	CUDA_CHECK(cudaDeviceSynchronize());

/* Write result image to file */
	write_device_texture_to_file((output_name + "_relaxed_cone_map_discrete.png").c_str(), cone_map);
}
