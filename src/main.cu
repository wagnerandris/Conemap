// STD
#include <cstdlib>
#include <filesystem>
#include <stdio.h>
#include <string>

// Boost
#include <boost/program_options.hpp>

#include "file_utils.cuh"
#include "kernels.cuh"

void convert_image(const char *filepath) {

  std::string output_name = std::filesystem::path(filepath).stem().string(); // TODO output path

/* Load image */
  unsigned char *d_input_image = nullptr;
  int width, height, channels;
  if (!read_texture_to_device(d_input_image, filepath, &width, &height, &channels))
    return;

  size_t size = width * height;
  // size_t input_size = size * channels;

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
                                        d_fod_dirs_image, d_fod, width, height,
                                        channels);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Write result image to file
  write_device_texture_to_file((output_name + "_fod.png").c_str(), d_fod_image,
                               width, height, 3);
  write_device_texture_to_file((output_name + "_fod_dirs.png").c_str(),
                               d_fod_dirs_image, width, height, 1);

/* Directional local maxima */
  // Allocate device memory
  unsigned char *d_local_max_dirs;
  CUDA_CHECK(cudaMalloc(&d_local_max_dirs, size));

  // Launch kernel
  local_max_dirs<<<blocks, threads>>>(d_input_image, d_local_max_dirs, width,
                                      height, channels);
  CUDA_CHECK(cudaDeviceSynchronize());

  // TODO local max dirs images


/* Relaxed cone map generation */
  // Allocate device memory
  unsigned char *d_cone_map;
  CUDA_CHECK(cudaMalloc(&d_cone_map, size * 4));

  // Launch kernel
  create_cone_map2<<<blocks, threads>>>(d_input_image, d_fod_image,
                                        d_local_max_dirs, d_cone_map, width,
                                        height, channels);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Write result image to file
  write_device_texture_to_file((output_name + "_relaxed_cone_map.png").c_str(),
                               d_cone_map, width, height, 4);


// /* Second order derivatives and watershed */
//   // Allocate device memory
//   unsigned char *d_sod_image, *d_watershed;
//   CUDA_CHECK(cudaMalloc(&d_sod_image, size * 4));
//   CUDA_CHECK(cudaMalloc(&d_watershed, size));
//
//   // Launch kernel
//   second_derivative<<<blocks, threads>>>(d_fod, d_sod_image, d_watershed, width,
//                                          height);
//   CUDA_CHECK(cudaDeviceSynchronize());
//
//   // Write result image to file
//   write_device_texture_to_file((output + "_sod.png").c_str(), d_sod_image,
//                                width, height, 4);
//   write_device_texture_to_file((output + "_watershed.png").c_str(), d_watershed,
//                                width, height, 1);
//
//   // Non maximum suppression
//   unsigned char *d_suppressed;
//   CUDA_CHECK(cudaMalloc(&d_suppressed, size));
//
//   // Launch kernel
//   non_maximum_suppression<<<blocks, threads>>>(d_input_image, d_fod_dirs,
//                                                d_watershed, d_suppressed, width,
//                                                height, channels);
//   CUDA_CHECK(cudaDeviceSynchronize());
//
//   // Write result image to file
//   write_device_texture_to_file((output + "_suppressed.png").c_str(),
//                                d_suppressed, width, height, 1);
//
//
// /* Relaxed cone map generation */
//   // Allocate device memory
//   unsigned char *d_cone_map;
//   CUDA_CHECK(cudaMalloc(&d_cone_map, size * 4));
//
//   // Launch kernel
//   create_cone_map<<<blocks, threads>>>(d_input_image, d_fod_image, d_suppressed,
//                                        d_cone_map, width, height, channels);
//   CUDA_CHECK(cudaDeviceSynchronize());
//
//   // Write result image to file
//   write_device_texture_to_file((output + "_relaxed_cone_map.png").c_str(),
//                                d_cone_map, width, height, 4);

/* Cleanup */
  CUDA_CHECK(cudaFree(d_input_image));
  CUDA_CHECK(cudaFree(d_fod_image));
  CUDA_CHECK(cudaFree(d_fod_dirs));
  CUDA_CHECK(cudaFree(d_fod_dirs_image));
  CUDA_CHECK(cudaFree(d_fod));
  // CUDA_CHECK(cudaFree(d_second_derivative_image));
  // CUDA_CHECK(cudaFree(d_watershed));
  // CUDA_CHECK(cudaFree(d_suppressed));
  CUDA_CHECK(cudaFree(d_local_max_dirs));
  CUDA_CHECK(cudaFree(d_cone_map));
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "No texture provided.\n");
    exit(0);
  }
  for (int i = 1; i < argc; ++i) {
    if (!std::filesystem::exists(argv[i]) &&
        std::filesystem::is_regular_file(argv[i])) {
      fprintf(stderr, "No such file: %s\n", argv[i]);
    }
    convert_image(argv[i]);
  }

  return 0;
}
