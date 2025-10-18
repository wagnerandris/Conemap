// STD
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

// Boost
#include <boost/program_options.hpp>

#include "file_utils.cuh"
#include "kernels.cuh"
#include "Texture.cuh"

struct SurfacePoint {
	int u;
	int v;
	unsigned char h;
};


bool find_neighbours_on_path(unsigned char* heightmap, unsigned char* watershed, int width, int height, SurfacePoint &current, std::vector<SurfacePoint> &neighbours) {
	for (int cv = std::max(current.v - 1, 0); cv <= std::min(current.v + 1, height - 1); ++cv) {
		for (int cu = std::max(current.u - 1, 0); cu <= std::min(current.u + 1, width - 1); ++cu) {
			int cidx = cv * width + cu;
			if (!watershed[cidx]) continue; // not a neighbour on path (current is already removed from watershed)

			if (heightmap[cidx] < current.h) return false; // current is not minimal

			neighbours.push_back(SurfacePoint{cu, cv, heightmap[cidx]});
		}
	}
	return true;
}

bool descend_on_path(unsigned char* heightmap, unsigned char* watershed, int width, int height, SurfacePoint &current) {
	SurfacePoint mp;
	unsigned char min = current.h;

	// find all valid neighbours
	for (int cv = std::max(current.v - 1, 0); cv <= std::min(current.v + 1, height - 1); ++cv) {
		for (int cu = std::max(current.u - 1, 0); cu <= std::min(current.u + 1, width - 1); ++cu) {
			int cidx = (cv * width + cu);
			unsigned char ch = heightmap[cidx];
			if (current.v * width + current.u == cidx || !watershed[cidx] || // not a neighbour on path
					ch >= min) // not lower than current min
				continue; 

			min = ch;
			mp = SurfacePoint{cu, cv, ch};
		}
	}

	if (min == current.h) return false; // haven't found anywhere to go

	current = mp;
	return true; // we need to continue
}

bool continue_path(unsigned char* heightmap, unsigned char* watershed, std::vector<SurfacePoint> &path, int width, int height) {
	SurfacePoint &current = path.back();
	SurfacePoint mp;
	int min = 256;

	// find all valid neighbours
	for (int cv = std::max(current.v - 1, 0); cv <= std::min(current.v + 1, height - 1); ++cv) {
		for (int cu = std::max(current.u - 1, 0); cu <= std::min(current.u + 1, width - 1); ++cu) {
			int cidx = (cv * width + cu);
			unsigned char ch = heightmap[cidx];
			if (!watershed[cidx] || // not a neighbour on path (current is already removed from watershed)
					ch < current.h || // lower than current
					ch >= min) // already found lower neighbour
				continue; 

			min = ch;
			mp = SurfacePoint{cu, cv, ch};
		}
	}

	if (min == 256) return false; // haven't found anywhere to go

	watershed[mp.v * width + mp.u] = 0;
	path.push_back(mp);
	return true; // we need to continue
}

void find_climbing_paths(unsigned char* heightmap, unsigned char* watershed, std::vector<std::vector<SurfacePoint>> &paths, int width, int height) {
	for (int u = 0; u < width; ++u) {
		for (int v = 0; v < height; ++v) {
			int idx = v * width + u;
			// repeat until the current texel is added to a path (and removed from watershed)
			while (watershed[idx]) {
				// starting from the current texel
				SurfacePoint current{u, v, heightmap[idx]};
				
				// descend on the steepest path
				while (descend_on_path(heightmap, watershed, width, height, current));
				watershed[current.v * width + current.u] = 0;

				// find all valid neighbours
				std::vector<SurfacePoint> neighbours;
				if (!find_neighbours_on_path(heightmap, watershed, width, height, current, neighbours)) continue; // if we weren't at a local minimum

				if (neighbours.size() == 0) {
					paths.push_back(std::vector<SurfacePoint>{current});
					continue;
				}

				// sort based on height
				std::sort(neighbours.begin(), neighbours.end(), [](SurfacePoint &a, SurfacePoint &b){return a.h < b.h;});

				// start a path in all directions not yet covered
				for (SurfacePoint neighbour : neighbours) {
					int nidx = neighbour.v * width + neighbour.u;
					if (!watershed[nidx]) continue;

					watershed[nidx] = 0;
					paths.push_back(std::vector<SurfacePoint>{current, neighbour});
					while (continue_path(heightmap, watershed, paths.back(), width, height));
				}
			}
				
		}
	}
}

void display_paths(const char* filename, std::vector<std::vector<SurfacePoint>> &paths, int width, int height) {
	int sum = 0;
	for (auto path : paths) {
		sum += path.size();
	}
	std::cout << "Texels: " << sum << '\n';
	std::cout << "Paths: " << paths.size() << '\n';
	std::cout << "Average texels per path: " << static_cast<float>(sum) / paths.size() << '\n';

	// TextureHostPointer<unsigned char> h_paths{width, height, 1};
	TextureHostPointer<unsigned char> h_paths{width, height, 3};

	for (auto path : paths) {
		uchar3 color;
		color.x = rand() % 256;
		color.y = rand() % 256;
		color.z = rand() % 256;
		for (SurfacePoint point : path) {
			// (*h_paths)[point.v * width + point.u] = 255;
			int idx = (point.v * width + point.u) * 3;
			(*h_paths)[idx]		 = color.x;
			(*h_paths)[idx + 1] = color.y;
			(*h_paths)[idx + 2] = color.z;
		}
	}

	write_host_texture_to_file(filename, h_paths);
}

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
//	// Allocate device memory
//	TextureDevicePointer<unsigned char> cone_map{width, height, 4};
//
// 	// Launch kernel
// 	create_cone_map_baseline<<<blocks, threads>>>(*input_image, *fod_image, *fod_exact_dirs, *watershed, *cone_map,
// 			width, height);
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
// 	// Write result image to file
// 	write_device_texture_to_file((output_name + "_relaxed_cone_map_baseline.png").c_str(), cone_map);
//
// /* Non maximum suppression */
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

	//	// Save local maxima in each direction to separate images
	// for (int i = 0; i < 8; ++i) {
	// 	bits_to_image<<<blocks, threads>>>(d_local_max_8dirs, *dir_bit_image,
	// 																		width, height, 1 << i);
	// 	CUDA_CHECK(cudaDeviceSynchronize());
	// 	write_device_texture_to_file((output_name + "_local_max_dir" + std::to_string(i) + ".png").c_str(),
	// 															 *dir_bit_image, width, height, 1);
	// }

	bits_to_image<<<blocks, threads>>>(*local_max_8dirs, *dir_bit_image,
																		width, height, 1);
	CUDA_CHECK(cudaDeviceSynchronize());

	// write_device_texture_to_file((output_name + "_local_max_dir1.png").c_str(), dir_bit_image);

	std::vector<std::vector<SurfacePoint>> paths;

	TextureHostPointer<unsigned char> h_heightmap{input_image};
	write_host_texture_to_file((output_name + "_heightmap.png").c_str(), h_heightmap);

	// dir1
	TextureHostPointer<unsigned char> h_watershed1{dir_bit_image};
	write_host_texture_to_file((output_name + "_local_max_dir1.png").c_str(), h_watershed1);
	
	find_climbing_paths(*h_heightmap, *h_watershed1, paths, width, height);

	display_paths((output_name + "dir1_climbing_paths.png").c_str(), paths, width, height);
	
	// 4dirs
	bits_to_image<<<blocks, threads>>>(*local_max_8dirs, *dir_bit_image,
																		width, height, 0b01010101);
	CUDA_CHECK(cudaDeviceSynchronize());
	TextureHostPointer<unsigned char> h_local_max_4dirs{dir_bit_image};
	write_host_texture_to_file((output_name + "_local_max_4dirs.png").c_str(), h_local_max_4dirs);
	
	paths.clear();
	find_climbing_paths(*h_heightmap, *h_local_max_4dirs, paths, width, height);

	display_paths((output_name + "local_max_4dir_climbing_paths.png").c_str(), paths, width, height);

	// 8dirs
	TextureHostPointer<unsigned char> h_local_max_8dirs{local_max_8dirs};
	write_host_texture_to_file((output_name + "_local_max_8dirs.png").c_str(), h_local_max_8dirs);
	
	paths.clear();
	find_climbing_paths(*h_heightmap, *h_local_max_8dirs, paths, width, height);

	display_paths((output_name + "local_max_8dir_climbing_paths.png").c_str(), paths, width, height);

//
// 	// Any of the 8
// 	bits_to_image<<<blocks, threads>>>(*local_max_8dirs, *dir_bit_image,
// 																		width, height, 0b11111111);
// 	CUDA_CHECK(cudaDeviceSynchronize());
// 	write_device_texture_to_file((output_name + "_local_max_8dirs.png").c_str(), dir_bit_image);
//
// 	// Any of the 4 axis aligned dirs
// 	bits_to_image<<<blocks, threads>>>(*local_max_8dirs, *dir_bit_image,
// 																		width, height, 0b01010101);
// 	CUDA_CHECK(cudaDeviceSynchronize());
// 	write_device_texture_to_file((output_name + "_local_max_4dirs.png").c_str(), dir_bit_image);
//
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
