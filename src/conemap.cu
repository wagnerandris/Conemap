// STD
#include <cstdint>
#include <filesystem>
#include <functional>

#include "conemap.hpp"
#include "file_utils.cuh"
#include "kernels.cuh"
#include "Texture.cuh"

class Conemap
{
public:
	Conemap(const std::filesystem::path output_folder, const std::filesystem::path filepath, const bool depthmap) :
		output_name(filepath.stem()),
		input_image(read_texture_to_device(filepath.c_str())),
		width(input_image.width), height(input_image.height), threads(8, 8),
		blocks((width + threads.x - 1) / threads.x,
					(height + threads.y - 1) / threads.y)
	{
		if (!input_image) return;

		// Handle depthmap
		if (depthmap) {
			output_name.append("_depthmap");
			invert<<<blocks, threads>>>(*input_image, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		output_path = output_folder / output_name;

	}

	std::filesystem::path run_analytic_generation()
	{
		TextureDevicePointer<uint8_t> cone_map(generate_analytic_packed_continuous());
	
		std::string analytic_output_name = output_name + "_relaxed_cone_map_analytic_sm.png";
		write_device_texture_to_file(analytic_output_name.c_str(), cone_map);

		return analytic_output_name;
	}
	
	std::filesystem::path run_discrete_generation()
	{
		TextureDevicePointer<uint8_t> cone_map(generate_compressed_continuous());

		std::string discrete_output_name = output_name + "_relaxed_cone_map_discrete.png";
		write_device_texture_to_file(discrete_output_name.c_str(), cone_map);

		return discrete_output_name;
	}

	void run_measurements()
	{
		printf("\n%s\n", output_name.c_str());
		measure("baseline", [this](){return generate_baseline();}, 3);
		measure("analytic_preprocess", [this](){return generate_analytic_preprocess();}, 3);
		measure("analytic_shared_memory", [this](){return generate_analytic_shared_memory();}, 3);
		measure("analytic_packed", [this](){return generate_analytic_packed();}, 3);
		measure("analytic_packed_continuous", [this](){return generate_analytic_packed_continuous();}, 3);
		measure("8dir_preprocess", [this](){return generate_8dir_preprocess();}, 3);
		measure("8dir_shared_memory", [this](){return generate_8dir_shared_memory();}, 3);
		measure("8dir_packed", [this](){return generate_8dir_packed();}, 3);
		measure("8dir_packed_continuous", [this](){return generate_8dir_packed_continuous();}, 3);
		measure("4dir_preprocess", [this](){return generate_4dir_preprocess();}, 3);
		measure("4dir_shared_memory", [this](){return generate_4dir_shared_memory();}, 3);
		measure("4dir_packed", [this](){return generate_4dir_packed();}, 3);
		measure("4dir_packed_continuous", [this](){return generate_4dir_packed_continuous();}, 3);
		measure("compressed", [this](){return generate_compressed();}, 3);
		measure("compressed_continuous", [this](){return generate_compressed_continuous();}, 3);
	}

private:
	std::string output_path;
	std::string output_name;
	TextureDevicePointer<uint8_t> input_image;
	const int width;
	const int height;
	const dim3 threads;
	const dim3 blocks;

	void measure(std::string type, std::function<TextureDevicePointer<uint8_t>()> generate_func, const uint measurement_num)
	{
		printf("\n%s\n", type.c_str());
		for (uint i = 0; i < measurement_num; ++i) {
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);

			cudaEventRecord(start);

			TextureDevicePointer<uint8_t> cone_map(generate_func());

			cudaEventRecord(stop);

			cudaEventSynchronize(stop);
			float milliseconds;
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("%f\n", milliseconds);

			if (i == measurement_num - 1) {
				write_device_texture_to_file((output_name + "_" + type + ".png").c_str(), cone_map);
			}
		}
	}
	
	TextureDevicePointer<uint8_t> generate_baseline()
	{
		/* Relaxed cone map generation: robust cone stepping, no preprocessing */
			// Allocate device memory
			TextureDevicePointer<uint8_t> cone_map{width, height, 4};

			// Launch kernel
			create_cone_map_baseline<<<blocks, threads>>>(*input_image, *cone_map, width, height);

			return cone_map;
	}

	TextureDevicePointer<uint8_t> generate_analytic_preprocess()
	{
		/* Relaxed cone map generation: continuous, analytic directions, no optimizations */
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

			// TODO delete?
			/* Non maximum suppression */
			// Allocate device memory
			TextureDevicePointer<bool> suppressed{width, height, 1};

			// Launch kernel
			non_maximum_suppression<<<blocks, threads>>>(*input_image, *fod_dirs, *watersheds, *suppressed, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			// Allocate device memory
			TextureDevicePointer<uint8_t> cone_map{width, height, 4};

			// Launch kernel
			create_cone_map_analytic<<<blocks, threads>>>(*input_image, *suppressed, *fod_dirs, *fods, *cone_map, width, height);

			return cone_map;
	}

	TextureDevicePointer<uint8_t> generate_analytic_shared_memory()
	{
		/* Relaxed cone map generation: continuous, analytic directions, shared memory */
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

			// TODO delete?
			/* Non maximum suppression */
			// Allocate device memory
			TextureDevicePointer<bool> suppressed{width, height, 1};

			// Launch kernel
			non_maximum_suppression<<<blocks, threads>>>(*input_image, *fod_dirs, *watersheds, *suppressed, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			// Allocate device memory
			TextureDevicePointer<uint8_t> cone_map{width, height, 4};

			// Launch kernel
			create_cone_map_analytic_shared_mem<<<blocks, threads>>>(*input_image, *suppressed, *fod_dirs, *fods, *cone_map, width, height);

			return cone_map;
	}

	TextureDevicePointer<uint8_t> generate_analytic_packed()
	{
		/* Relaxed cone map generation: continuous, analytic directions, packed shared memory */
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

			// TODO delete?
			/* Non maximum suppression */
			// Allocate device memory
			TextureDevicePointer<bool> suppressed{width, height, 1};

			// Launch kernel
			non_maximum_suppression<<<blocks, threads>>>(*input_image, *fod_dirs, *watersheds, *suppressed, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			/* Pack data */
			// Allocate device memory
			TextureDevicePointer<AnalyticData> packed{width, height, 1};

			// Launch kernel
			pack<<<blocks, threads>>>(*input_image, *suppressed, *fod_dirs, *packed, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());


			// Allocate device memory
			TextureDevicePointer<uint8_t> cone_map{width, height, 4};

			// Launch kernel
			create_cone_map_analytic_packed<<<blocks, threads>>>(*input_image, *packed, *fods, *cone_map, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			return cone_map;
	}
	
	TextureDevicePointer<uint8_t> generate_analytic_packed_continuous()
	{
		/* Relaxed cone map generation: continuous, analytic directions, continuously packed shared memory */
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

			// TODO delete?
			/* Non maximum suppression */
			// Allocate device memory
			TextureDevicePointer<bool> suppressed{width, height, 1};

			// Launch kernel
			non_maximum_suppression<<<blocks, threads>>>(*input_image, *fod_dirs, *watersheds, *suppressed, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			/* Pack data */
			// Allocate device memory
			TextureDevicePointer<IndexedAnalyticData> packed{width, height, 1};

			// Launch kernel
			pack<<<blocks, threads>>>(*input_image, *suppressed, *fod_dirs, *packed, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			// Allocate device memory
			TextureDevicePointer<uint8_t> cone_map{width, height, 4};

			// Launch kernel
			create_cone_map_analytic_continuous<<<blocks, threads>>>(*input_image, *packed, *fods, *cone_map, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			return cone_map;
	}

	TextureDevicePointer<uint8_t> generate_8dir_preprocess()
	{
		/* Relaxed cone map generation: 8 discrete directions, no optimizations */
			// Allocate device memory
			TextureDevicePointer<uint8_t> local_max_8dirs{width, height, 1};

			// Launch kernel
			local_max_8dir<<<blocks, threads>>>(*input_image, *local_max_8dirs, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			// Allocate device memory
			TextureDevicePointer<uint8_t> cone_map{width, height, 4};

			// Launch kernel
			create_cone_map_8dir<<<blocks, threads>>>(*input_image, *local_max_8dirs, *cone_map, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			return cone_map;
	}

	TextureDevicePointer<uint8_t> generate_8dir_shared_memory()
	{
		/* Relaxed cone map generation: 8 discrete directions, shared memory */
			// Allocate device memory
			TextureDevicePointer<uint8_t> local_max_8dirs{width, height, 1};

			// Launch kernel
			local_max_8dir<<<blocks, threads>>>(*input_image, *local_max_8dirs, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			// Allocate device memory
			TextureDevicePointer<uint8_t> cone_map{width, height, 4};

			// Launch kernel
			create_cone_map_8dir_shared_mem<<<blocks, threads>>>(*input_image, *local_max_8dirs, *cone_map, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			return cone_map;
	}

	TextureDevicePointer<uint8_t> generate_8dir_packed()
	{
		/* Relaxed cone map generation: 8 discrete directions, packed shared memory */
			// Allocate device memory
			TextureDevicePointer<DiscreteData8Dirs> packed{width, height, 1};

			// Launch kernel
			pack_discrete<DiscreteData8Dirs><<<blocks, threads>>>(*input_image, *packed, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());
			
			// Allocate device memory
			TextureDevicePointer<uint8_t> cone_map{width, height, 4};
			
			// Launch kernel
			create_cone_map_discrete_packed<DiscreteData8Dirs><<<blocks, threads>>>(*input_image, *packed, *cone_map, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			return cone_map;
	}

	TextureDevicePointer<uint8_t> generate_8dir_packed_continuous()
	{
		/* Relaxed cone map generation: 8 discrete directions, continuously packed shared memory */
			// Allocate device memory
			TextureDevicePointer<IndexedDiscreteData8Dirs> packed{width, height, 1};

			// Launch kernel
			pack_discrete_continuously<IndexedDiscreteData8Dirs><<<blocks, threads>>>(*input_image, *packed, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());
			
			// Allocate device memory
			TextureDevicePointer<uint8_t> cone_map{width, height, 4};
			
			// Launch kernel
			create_cone_map_discrete_continuous<IndexedDiscreteData8Dirs><<<blocks, threads>>>(*input_image, *packed, *cone_map, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			return cone_map;
	}

	TextureDevicePointer<uint8_t> generate_4dir_preprocess()
	{
		/* Relaxed cone map generation: 8 discrete directions, no optimizations */
			// Allocate device memory
			TextureDevicePointer<uint8_t> local_max_4dirs{width, height, 1};

			// Launch kernel
			local_max_4dir<<<blocks, threads>>>(*input_image, *local_max_4dirs, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			// Allocate device memory
			TextureDevicePointer<uint8_t> cone_map{width, height, 4};

			// Launch kernel
			create_cone_map_4dir<<<blocks, threads>>>(*input_image, *local_max_4dirs, *cone_map, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			return cone_map;
	}

	TextureDevicePointer<uint8_t> generate_4dir_shared_memory()
	{
		/* Relaxed cone map generation: 8 discrete directions, shared memory */
			// Allocate device memory
			TextureDevicePointer<uint8_t> local_max_4dirs{width, height, 1};

			// Launch kernel
			local_max_4dir<<<blocks, threads>>>(*input_image, *local_max_4dirs, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			// Allocate device memory
			TextureDevicePointer<uint8_t> cone_map{width, height, 4};

			// Launch kernel
			create_cone_map_4dir_shared_mem<<<blocks, threads>>>(*input_image, *local_max_4dirs, *cone_map, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			return cone_map;
	}

	TextureDevicePointer<uint8_t> generate_4dir_packed()
	{
		/* Relaxed cone map generation: 4 discrete directions, packed shared memory */
			// Allocate device memory
			TextureDevicePointer<DiscreteData4Dirs> packed{width, height, 1};

			// Launch kernel
			pack_discrete<DiscreteData4Dirs><<<blocks, threads>>>(*input_image, *packed, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());
			
			// Allocate device memory
			TextureDevicePointer<uint8_t> cone_map{width, height, 4};
			
			// Launch kernel
			create_cone_map_discrete_packed<DiscreteData4Dirs><<<blocks, threads>>>(*input_image, *packed, *cone_map, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			return cone_map;
	}

	TextureDevicePointer<uint8_t> generate_4dir_packed_continuous()
	{
		/* Relaxed cone map generation: 4 discrete directions, continuously packed shared memory */
			// Allocate device memory
			TextureDevicePointer<IndexedDiscreteData4Dirs> packed{width, height, 1};

			// Launch kernel
			pack_discrete_continuously<IndexedDiscreteData4Dirs><<<blocks, threads>>>(*input_image, *packed, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());
			
			// Allocate device memory
			TextureDevicePointer<uint8_t> cone_map{width, height, 4};
			
			// Launch kernel
			create_cone_map_discrete_continuous<IndexedDiscreteData4Dirs><<<blocks, threads>>>(*input_image, *packed, *cone_map, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			return cone_map;
	}

	TextureDevicePointer<uint8_t> generate_compressed()
	{
		/* Relaxed cone map generation: 4 discrete directions, compressed, shared memory */
			// Allocate device memory
			TextureDevicePointer<uint16_t> compressed{width, height, 1};

			// Launch kernel
			pack_discrete<<<blocks, threads>>>(*input_image, *compressed, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());
			
			// Allocate device memory
			TextureDevicePointer<uint8_t> cone_map{width, height, 4};
			
			// Launch kernel
			create_cone_map_discrete_packed<uint16_t><<<blocks, threads>>>(*input_image, *compressed, *cone_map, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			return cone_map;
	}

	TextureDevicePointer<uint8_t> generate_compressed_continuous()
	{
		/* Relaxed cone map generation: 4 discrete directions, compressed, continuously packed shared memory */
			// Allocate device memory
			TextureDevicePointer<uint16_t> compressed{width, height, 1};

			// Launch kernel
			pack_discrete_continuously<uint16_t><<<blocks, threads>>>(*input_image, *compressed, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());
			
			// Allocate device memory
			TextureDevicePointer<uint8_t> cone_map{width, height, 4};
			
			// Launch kernel
			create_cone_map_discrete_continuous<uint16_t><<<blocks, threads>>>(*input_image, *compressed, *cone_map, width, height);
			CUDA_CHECK(cudaDeviceSynchronize());

			return cone_map;
	}
};


void conemap::measure(const std::filesystem::path output_path, const std::filesystem::path filepath, const bool depthmap) {
	Conemap(output_path, filepath, depthmap).run_measurements();
}
std::filesystem::path conemap::analytic(const std::filesystem::path output_path, const std::filesystem::path filepath, const bool depthmap) {
	return Conemap(output_path, filepath, depthmap).run_analytic_generation();
}
std::filesystem::path conemap::discrete(const std::filesystem::path output_path, const std::filesystem::path filepath, const bool depthmap) {
	return Conemap(output_path, filepath, depthmap).run_discrete_generation();
}
