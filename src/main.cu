// STD
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

// Boost
#include <boost/program_options.hpp>

#include "file_utils.cuh"
#include "kernels.cuh"

static std::filesystem::path output_path;

void convert_image(const char *filepath) {

	std::string output_name = output_path / std::filesystem::path(filepath).stem();

/* Load image */
  unsigned char *d_input_image = nullptr;
  int width, height, channels;
  if (!read_texture_to_device(d_input_image, filepath, &width, &height, &channels))
    return;

  size_t size = width * height;


  // TODO delete
  write_device_texture_to_file((output_name + ".png").c_str(), d_input_image, width, height, 1);

  // Threads/blocks
  // TODO what's optimal?
  dim3 threads(16, 16);
  dim3 blocks((width + threads.x - 1) / threads.x,
              (height + threads.y - 1) / threads.y);

/* First order derivatives */
  // Allocate device memory
  int *d_fod;
  unsigned char *d_fod_image, *d_fod_dirs, *d_fod_dirs_image;

  CUDA_CHECK(cudaMalloc(&d_fod, size * 2 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_fod_image, size * 3));
  CUDA_CHECK(cudaMalloc(&d_fod_dirs, size));
  CUDA_CHECK(cudaMalloc(&d_fod_dirs_image, size));

  // Launch kernel
  first_derivative<<<blocks, threads>>>(d_input_image, d_fod_image, d_fod_dirs,
                                        d_fod_dirs_image, d_fod, width, height);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Write result image to file
  write_device_texture_to_file((output_name + "_fod.png").c_str(), d_fod_image,
                               width, height, 3);
  write_device_texture_to_file((output_name + "_fod_dirs.png").c_str(),
                               d_fod_dirs_image, width, height, 1);

/* Directional local maxima */
  // Allocate device memory
  unsigned char *d_local_max_8dirs, *d_dir_bit_image;
  CUDA_CHECK(cudaMalloc(&d_local_max_8dirs, size));
  CUDA_CHECK(cudaMalloc(&d_dir_bit_image, size));

  // Launch kernel
  local_max_8dirs<<<blocks, threads>>>(d_input_image, d_local_max_8dirs,
																			 width, height);
  CUDA_CHECK(cudaDeviceSynchronize());

	//  // Save local maxima in each direction to separate images
	// for (int i = 0; i < 8; ++i) {
	// 	bits_to_image<<<blocks, threads>>>(d_local_max_8dirs, d_dir_bit_image,
	// 																		width, height, 1 << i);
	// 	CUDA_CHECK(cudaDeviceSynchronize());
	// 	write_device_texture_to_file((output_name + "_local_max_dir" + std::to_string(i) + ".png").c_str(),
	// 															 d_dir_bit_image, width, height, 1);
	// }

	// Any of the 8
	bits_to_image<<<blocks, threads>>>(d_local_max_8dirs, d_dir_bit_image,
																		width, height, 0b11111111);
  CUDA_CHECK(cudaDeviceSynchronize());
	write_device_texture_to_file((output_name + "_local_max_8dirs.png").c_str(),
															 d_dir_bit_image, width, height, 1);

	// Any of the 4 axis aligned dirs
	bits_to_image<<<blocks, threads>>>(d_local_max_8dirs, d_dir_bit_image,
																		width, height, 0b01010101);
  CUDA_CHECK(cudaDeviceSynchronize());
	write_device_texture_to_file((output_name + "_local_max_4dirs.png").c_str(),
															 d_dir_bit_image, width, height, 1);


// /* Create binary mipmaps */
//   int mipmap_width  = (width  + 1) / 2;
//   int mipmap_height = (height + 1) / 2;
//
//   unsigned char *d_binary_mipmap;
//   CUDA_CHECK(cudaMalloc(&d_binary_mipmap, mipmap_width * mipmap_height));
//
// 	create_binary_mipmap_level<<<blocks, threads>>>(d_binary_mipmap, d_local_max_8dirs, width, height, mipmap_width, mipmap_height);
// 	CUDA_CHECK(cudaDeviceSynchronize());
//
//   CUDA_CHECK(cudaFree(d_dir_bit_image));
//   CUDA_CHECK(cudaMalloc(&d_dir_bit_image, mipmap_width * mipmap_height));
//
// 	// Any of the 8
// 	bits_to_image<<<blocks, threads>>>(d_binary_mipmap, d_dir_bit_image,
// 																		mipmap_width, mipmap_height, 0b11111111);
//   CUDA_CHECK(cudaDeviceSynchronize());
// 	write_device_texture_to_file((output_name + "_local_max_8dirs_mipmap.png").c_str(),
// 															 d_dir_bit_image, mipmap_width, mipmap_height, 1);
//
// 	// Any of the 4 axis aligned dirs
// 	bits_to_image<<<blocks, threads>>>(d_binary_mipmap, d_dir_bit_image,
// 																		mipmap_width, mipmap_height, 0b01010101);
//   CUDA_CHECK(cudaDeviceSynchronize());
// 	write_device_texture_to_file((output_name + "_local_max_4dirs_mipmap.png").c_str(),
// 															 d_dir_bit_image, mipmap_width, mipmap_height, 1);

/* Relaxed cone map generation */
  // Allocate device memory
  unsigned char *d_cone_map;
  CUDA_CHECK(cudaMalloc(&d_cone_map, size * 4));

  // Launch kernel
  create_cone_map_8dirs<<<blocks, threads>>>(d_input_image, d_fod_image,
																						 d_local_max_8dirs, d_cone_map,
																						 width, height);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Write result image to file
  write_device_texture_to_file((output_name + "_relaxed_cone_map_8dirs.png").c_str(),
                               d_cone_map, width, height, 4);

  // Launch kernel
  create_cone_map_4dirs<<<blocks, threads>>>(d_input_image, d_fod_image,
																						 d_local_max_8dirs, d_cone_map,
																						 width, height);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Write result image to file
  write_device_texture_to_file((output_name + "_relaxed_cone_map_4dirs.png").c_str(),
                               d_cone_map, width, height, 4);


/* Second order derivatives and watershed */
  // Allocate device memory
  unsigned char *d_sod_image, *d_watershed;
  CUDA_CHECK(cudaMalloc(&d_sod_image, size * 4));
  CUDA_CHECK(cudaMalloc(&d_watershed, size));

  // Launch kernel
  second_derivative<<<blocks, threads>>>(d_fod, d_sod_image, d_watershed,
																				 width, height);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Write result image to file
  write_device_texture_to_file((output_name + "_sod.png").c_str(), d_sod_image,
                               width, height, 4);
  write_device_texture_to_file((output_name + "_watershed.png").c_str(), d_watershed,
                               width, height, 1);

/* Non maximum suppression */
  // Allocate device memory
  unsigned char *d_suppressed;
  CUDA_CHECK(cudaMalloc(&d_suppressed, size));

  // Launch kernel
  non_maximum_suppression<<<blocks, threads>>>(d_input_image, d_fod_dirs,
                                               d_watershed, d_suppressed,
                                               width, height);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Write result image to file
  write_device_texture_to_file((output_name + "_suppressed.png").c_str(),
                               d_suppressed, width, height, 1);


/* Relaxed cone map generation */
  // Launch kernel
  create_cone_map_analytic<<<blocks, threads>>>(d_input_image, d_fod_image, d_fod_dirs, d_suppressed,
																								d_cone_map, width, height);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Write result image to file
  write_device_texture_to_file((output_name + "_relaxed_cone_map_analytic.png").c_str(),
                               d_cone_map, width, height, 4);

/* Cleanup */
  CUDA_CHECK(cudaFree(d_input_image));
  CUDA_CHECK(cudaFree(d_fod));
  CUDA_CHECK(cudaFree(d_fod_image));
  CUDA_CHECK(cudaFree(d_fod_dirs));
  CUDA_CHECK(cudaFree(d_fod_dirs_image));
  CUDA_CHECK(cudaFree(d_sod_image));
  CUDA_CHECK(cudaFree(d_watershed));
  CUDA_CHECK(cudaFree(d_suppressed));
  CUDA_CHECK(cudaFree(d_local_max_8dirs));
  CUDA_CHECK(cudaFree(d_dir_bit_image));
  // CUDA_CHECK(cudaFree(d_binary_mipmap));
  CUDA_CHECK(cudaFree(d_cone_map));
}

int main(int argc, char* argv[]) {
  std::vector<std::string> input_files;

  // Possible options
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("output,o", boost::program_options::value<std::filesystem::path>(&output_path)->default_value("."), "set path to output folder")
    ("input,i", boost::program_options::value<std::vector<std::string>>(&input_files), "input files");
		//TODO flip Y, depthmap, wrap

  // Positional options
  boost::program_options::positional_options_description pod;
  pod.add("input", -1);  // all remaining options

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
    std::cout << "Usage: " << argv[0] << " [-o OUTPUT] INPUT [INPUT]...\n";
    std::cout << desc << "\n";
    return 0;
  }

	// No input
  if (input_files.empty()) {
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
  for (auto file : input_files) {
    if (!std::filesystem::exists(file) || !std::filesystem::is_regular_file(file)) {
			std::cerr << "Error: " << file << " is not a file.\n";
			continue;
    }
    convert_image(file.c_str());
  }

  return 0;
}
