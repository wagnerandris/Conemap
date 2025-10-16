// STD
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

// Boost
#include <boost/program_options.hpp>

#include "file_utils.cuh"
#include "kernels.cuh"
#include "Texture.cuh"
#include "Mipmap.cuh"

static std::filesystem::path output_path;

void convert_image(const char *filepath, bool depthmap = false) {

	std::string output_name = depthmap ?
		output_path / std::filesystem::path(filepath).stem().concat("_depthmap") :
		output_path / std::filesystem::path(filepath).stem();

/* Load image */
	TextureDevicePointer<unsigned char> input_image = read_texture_to_device(filepath);
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

/* Max mipmaps */
	MipmapDevicePointer<unsigned char> max_mipmaps(input_image, &create_max_mipmap_level);

	// For all mipmap levels
	for (size_t l = 0; l < max_mipmaps.mipmap_levels.size(); ++l) {
		TextureDevicePointer<unsigned char>* level = max_mipmaps.mipmap_levels[l];

		// Save
		write_device_texture_to_file((output_name + "_max_mip_level" + std::to_string(l) + ".png").c_str(), *level);
	}
//
// /* First order derivatives */
// 	// Allocate device memory
// 	TextureDevicePointer<int> fods{width, height, 2};
// 	TextureDevicePointer<float> fod_exact_dirs{width, height, 1};
// 	TextureDevicePointer<unsigned char>
// 		fod_image{width, height, 3},
// 		fod_discrete_dirs{width, height, 1},
// 		fod_dirs_image{width, height, 1};
//
// 	// Launch kernel
// 	fod<<<blocks, threads>>>(*input_image, *fods, *fod_image, *fod_exact_dirs, *fod_discrete_dirs, *fod_dirs_image, width, height);
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	// Write result image to file
// 	write_device_texture_to_file((output_name + "_fod.png").c_str(), fod_image);
// 	write_device_texture_to_file((output_name + "_fod_dirs.png").c_str(), fod_dirs_image);
//
// /* Second order derivatives and watershed */
// 	// Allocate device memory
// 	TextureDevicePointer<unsigned char>
// 		sod_image{width, height, 4},
// 		watershed{width, height, 1};
//
// 	// Launch kernel
// 	sod_and_watershed<<<blocks, threads>>>(*fods, *sod_image, *watershed,
// 																				 width, height);
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	// Write result image to file
// 	write_device_texture_to_file((output_name + "_sod.png").c_str(), sod_image);
// 	write_device_texture_to_file((output_name + "_watershed.png").c_str(), watershed);
//
// /* Non maximum suppression */
// 	// Allocate device memory
// 	TextureDevicePointer<unsigned char> suppressed{width, height, 1};
//
// 	// Launch kernel
// 	non_maximum_suppression<<<blocks, threads>>>(*input_image, *fod_discrete_dirs,
// 																							 *watershed, *suppressed,
// 																							 width, height);
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	// Write result image to file
// 	write_device_texture_to_file((output_name + "_suppressed.png").c_str(), suppressed);
//
// /* Relaxed cone map generation: Baseline */
// 	// Allocate device memory
// 	TextureDevicePointer<unsigned char> cone_map{width, height, 4};
//
// 	// Launch kernel
// 	create_cone_map_baseline<<<blocks, threads>>>(*input_image, *fod_image, *fod_exact_dirs, *watershed, *cone_map,
// 			width, height);
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	// Write result image to file
// 	write_device_texture_to_file((output_name + "_relaxed_cone_map_baseline.png").c_str(), cone_map);
//
// /* Relaxed cone map generation: Analytic */
// 	// Launch kernel
// 	create_cone_map_analytic<<<blocks, threads>>>(*input_image, *fod_image, *fod_discrete_dirs, *suppressed,
// 																								*cone_map, width, height);
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	// Write result image to file
// 	write_device_texture_to_file((output_name + "_relaxed_cone_map_analytic.png").c_str(), cone_map);

/* Directional local maxima */
	// Allocate device memory
	TextureDevicePointer<unsigned char>
		local_max_8dirs{width, height, 1},
		dir_bit_image{width, height, 1};

	// Launch kernel
	local_max_8dir<<<blocks, threads>>>(*input_image, *local_max_8dirs,
																			 width, height);
	CUDA_CHECK(cudaDeviceSynchronize());

/* Binary mipmaps */
	MipmapDevicePointer<unsigned char> local_max_8dirs_mipmaps(local_max_8dirs, &create_binary_mipmap_level);

	// For all mipmap levels
	for (size_t l = 0; l < local_max_8dirs_mipmaps.mipmap_levels.size(); ++l) {
		TextureDevicePointer<unsigned char>* level = local_max_8dirs_mipmaps.mipmap_levels[l];
		TextureDevicePointer<unsigned char> level_image{level->width, level->height, level->channels};

		// Save local maxima in each direction to separate images
		for (int i = 0; i < 8; ++i) {
			// Create image
			bits_to_image<<<blocks, threads>>>(**level, *level_image,
																				level->width, level->height, 1 << i);
			CUDA_CHECK(cudaDeviceSynchronize());

			// Save
			write_device_texture_to_file((output_name + "_local_max_dir" + std::to_string(i) + "_mip_level" + std::to_string(l) + ".png").c_str(), level_image);
		}
	}

	exit(0);

	// // Any of the 8
	// bits_to_image<<<blocks, threads>>>(*local_max_8dirs, *dir_bit_image,
	// 																	width, height, 0b11111111);
	// CUDA_CHECK(cudaDeviceSynchronize());
	// write_device_texture_to_file((output_name + "_local_max_8dirs.png").c_str(), dir_bit_image);
	//
	// // Any of the 4 axis aligned dirs
	// bits_to_image<<<blocks, threads>>>(*local_max_8dirs, *dir_bit_image,
	// 																	width, height, 0b01010101);
	// CUDA_CHECK(cudaDeviceSynchronize());
	// write_device_texture_to_file((output_name + "_local_max_4dirs.png").c_str(), dir_bit_image);
	//

// /* Relaxed cone map generation: Discrete directions */
// 	// Launch kernel
// 	create_cone_map_8dir<<<blocks, threads>>>(*input_image, *fod_image,
// 																						 *local_max_8dirs, *cone_map,
// 																						 width, height);
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	// Write result image to file
// 	write_device_texture_to_file((output_name + "_relaxed_cone_map_8dirs.png").c_str(), cone_map);
//
// 	// Launch kernel
// 	create_cone_map_4dir<<<blocks, threads>>>(*input_image, *fod_image,
// 																						 *local_max_8dirs, *cone_map,
// 																						 width, height);
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	// Write result image to file
// 	write_device_texture_to_file((output_name + "_relaxed_cone_map_4dirs.png").c_str(), cone_map);
}

int main(int argc, char* argv[]) {
	std::vector<std::string> heightmap_files;
	std::vector<std::string> depthmap_files;

	// Possible options
	boost::program_options::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "produce help message")
		("output,o", boost::program_options::value<std::filesystem::path>(&output_path)->default_value("."), "set path to output folder")
		("heightmap", boost::program_options::value<std::vector<std::string>>(&heightmap_files), "input heightmap file")
		("depthmap,d", boost::program_options::value<std::vector<std::string>>(&depthmap_files), "input depthmap file");

	// Positional options
	boost::program_options::positional_options_description pod;
	pod.add("heightmap", -1);	// all remaining options

	boost::program_options::variables_map vm;

	try {
		boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(pod).run(), vm);
		boost::program_options::notify(vm);
	} catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << "\n";
		return 1;
	}

	// Help
	if (vm.count("help")) {
		std::cout << "Usage: " << argv[0] << " [-o OUTPUT] [-d] INPUT [[-d] INPUT]...\n";
		std::cout << desc << "\n";
		return 0;
	}

	// No input
	if (heightmap_files.empty() && depthmap_files.empty()) {
		std::cerr << "Error: No input files provided.\n";
		return 1;
	}

	// Output
	if (!std::filesystem::exists(output_path)) {
		std::filesystem::create_directory(output_path);
	} else if (!std::filesystem::is_directory(output_path)) {
		std::cerr << "Error: " << output_path << " is not a directory.\n";
		return 1;
	}

	// OK
	for (auto file : heightmap_files) {
		if (!std::filesystem::exists(file) || !std::filesystem::is_regular_file(file)) {
			std::cerr << "Error: " << file << " is not a file.\n";
			continue;
		}
		convert_image(file.c_str());
	}

	// OK
	for (auto file : depthmap_files) {
		if (!std::filesystem::exists(file) || !std::filesystem::is_regular_file(file)) {
			std::cerr << "Error: " << file << " is not a file.\n";
			continue;
		}
		convert_image(file.c_str(), true);
	}

	return 0;
}
