#pragma once

// STD
#include <sys/types.h>
#include <vector>
#include <string>

#include "file_utils.cuh"
#include "Texture.cuh"

template <typename T>
struct MipmapDevicePointer {
	std::vector<TextureDevicePointer<T> *> mipmap_levels;

	MipmapDevicePointer(TextureDevicePointer<T> &original,
											void (mipmap_func)(T*, int, int, T*, int, int)) // Requires kernel function that makes consecutive leves
	{
		mipmap_levels.push_back(&original);

  	int current_width  = original.width;
  	int current_height = original.height;
  	int channels = original.channels;

		while(current_width > 1 && current_height > 1) {
			// Half the side lengths
  		current_width  = (current_width  + 1) / 2;
  		current_height = (current_height + 1) / 2;

			// Create next level
  		TextureDevicePointer<T> *current_level = new TextureDevicePointer<T>{current_width, current_height, channels};

			// Threads/blocks
			// TODO what's optimal?
			dim3 threads(16, 16);
			dim3 blocks((current_width + threads.x - 1) / threads.x,
									(current_height + threads.y - 1) / threads.y);

			// Call kernel function to fill level
			mipmap_func<<<blocks, threads>>>(**mipmap_levels.back(), mipmap_levels.back()->width, mipmap_levels.back()->height, **current_level, current_width, current_height);
			CUDA_CHECK(cudaDeviceSynchronize());

			// Add to vector
			mipmap_levels.push_back(current_level);
			
			// write_device_texture_to_file(("test_mipmap/local_max_mip_level" + std::to_string(mipmap_levels.size()) + ".png").c_str(), *current_level);
		}
	}


	/*
	cudaTextureObject_t to_texture() {
		// Create mipmapped array
		cudaMipmappedArray_t mipmapped_array;
		cudaChannelFormatDesc channel_format = cudaChannelFormatDesc{8, 0, 0, 0, cudaChannelFormatKindUnsigned};
		cudaExtent extent = cudaExtent{static_cast<unsigned long>(mipmap_levels.begin()->width), static_cast<unsigned long>(mipmap_levels.begin()->height)};
		CUDA_CHECK(cudaMallocMipmappedArray(&mipmapped_array, &channel_format, extent, mipmap_levels.size()));
		
		// Copy each level to the corresponding array
		for (size_t i = 0; i < mipmap_levels.size(); ++i) {
			TextureDevicePointer<T>* current_level = mipmap_levels[i];

			cudaArray_t level_array;
			CUDA_CHECK(cudaGetMipmappedArrayLevel(&level_array, mipmapped_array, i))
			
			int pitch = current_level->width * current_level->channels * sizeof(T);
			CUDA_CHECK(cudaMemcpy2DToArray(
				level_array, // destination
    		0, 0, // offset
    		**current_level, // source
    		pitch, // pitch
    		pitch, // width in bytes
    		current_level->height, // height
    		cudaMemcpyDeviceToDevice
			));
		}

		cudaResourceDesc resourcde_desc = {};
		resourcde_desc.resType = cudaResourceTypeMipmappedArray;
		resourcde_desc.res.mipmap.mipmap = mipmapped_array;

		cudaTextureDesc texture_desc = {};
		texture_desc.addressMode[0] = cudaAddressModeWrap;       // or Border/Wrap
		texture_desc.addressMode[1] = cudaAddressModeWrap;
		texture_desc.filterMode     = cudaFilterModePoint;        // No interpolation
		texture_desc.readMode       = cudaReadModeNormalizedFloat; // or cudaReadModeElementType
		texture_desc.normalizedCoords = 0;                        // Integer coordinates
		texture_desc.mipmapFilterMode = cudaFilterModePoint;      // For mipmapping (optional)

		cudaTextureObject_t texObj;
		CUDA_CHECK(cudaCreateTextureObject(&texObj, &resourcde_desc, &texture_desc, nullptr));
		return texObj;
	}
	*/

	~MipmapDevicePointer() {
		for (size_t i = 1; i < mipmap_levels.size(); ++i) {
			delete mipmap_levels[i]; // do not delete the original one
		}
	}
};
