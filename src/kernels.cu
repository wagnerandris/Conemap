#include "kernels.cuh"


__global__ void local_max_8dirs(unsigned char* heightmap, unsigned char* dirs, int width, int height, int channels) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;

  if (u >= width || v >= height) return;

	int h = heightmap[(v * width + u) * channels];

    // Direction vectors: Right, Top-Right, Top, Top-Left, Left, Bottom-Left, Bottom, Bottom-Right
    int du[8] = { 1, 1, 0, -1, -1, -1, 0, 1 };
    int dv[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

    unsigned char result = 0;

    for (int dir = 0; dir < 8; ++dir) {
        int un = u + du[dir];
        int up = u - du[dir];
        int vn = v + dv[dir];
        int vp = v - dv[dir];

        // Ensure both neighbors are within bounds
        bool u_good = (un >= 0 && un < width && vn >= 0 && vn < height);
        bool v_good = (up >= 0 && up < width && vp >= 0 && vp < height);

				// edges are always max
				// TODO looped textures?
        int hn = u_good ? heightmap[(vn * width + un) * channels] : -1;
        int hp = v_good ? heightmap[(vp * width + up) * channels] : -1;

				// local max in hp to hn direction (if there is a plateau, we need its last point)
        if (h > hn && h >= hp) {
            result |= (1 << dir);
        }
    }

    dirs[v * width + u] = result;
}

__global__ void bits_to_image(unsigned char* input, unsigned char* output_image, int width, int height, unsigned char bitmask) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;

  if (u >= width || v >= height) return;

  output_image[v* width + u] = input[v * width + u] & bitmask ? 255 : 0;
}

__global__ void create_cone_map_8dirs(unsigned char* heightmap, unsigned char* derivative_image, unsigned char* dirs, unsigned char* cone_map, int width, int height, int channels) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;

  if (u >= width || v >= height) return;

	float iwidth = 1.0f / width;
	float iheight = 1.0f / height;

	// TODO Why are we assuming run/rise = 1 exactly (instead of infinity)? Float textures?
	float min_ratio2 = 1.0f;

	// normalize height
	float h = heightmap[(v * width + u) * channels] / 255.0f;

	// init variables
	int du, dv;
	float dun, dvn;
	int start, end;

	// search in an increasing radius spiral around the texel
	for (int rad = 1;
			// TODO why the 1.1f?
			// otherwise we can't find anything steeper than the current min_ratio (see min_ratio assignment)
			rad * rad <= 1.1f * (1.0f - h) * width *
									 1.1f * (1.0f - h) * height *
									 min_ratio2 &&
			// because we started from 1, and further than (1.0f - h) * width, we couldn't find anything steeper than 1
			rad <= 1.1f * (1.0f - h) * width &&
			rad <= 1.1f * (1.0f - h) * height;
			++rad) {

		// Right side

		// u displacement
		du = u + rad;
		// normalized
		dun = rad * iwidth;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (du >= width) du -= width; 
		// set v limits
		start = max(v - rad, 0);
		end = min(v + rad - 1, height);

		// go through side
		for (int dv = start; dv < end; ++dv) {
			// check if local maxima in the given direction, skip if not

			int discrete_dir;
			if (dv - start <= rad / 2) discrete_dir = 1;
			else if (end - dv <= rad / 2) discrete_dir = 7;
			else discrete_dir = 0;

			// float dir = atan2f(dv, du);
			// unsigned char discrete_dir = static_cast<unsigned char>(
			// 													(dir + M_PI // all positive
			// 													+ M_PI_4f / 2.0f) // align regions
			// 													/ M_PI_4f); // 8 dirs
			if (!(dirs[(dv * width + du)] & 1 << discrete_dir)) continue;

			// normalize v displacement
			dvn = (dv - v) * iheight;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}

		// Top side

		// u displacement
		dv = v - rad;
		// normalized
		dvn = -rad * iheight;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (dv < 0) dv += height; 
		// set u limits
		start = max(u - rad, 0);
		end = min(u + rad - 1, width);

		// go through side
		for (int du = start; du < end; ++du) {
			// check if local maxima in the given direction, skip if not

			int discrete_dir;
			if (du - start <= rad / 2) discrete_dir = 3;
			else if (end - du <= rad / 2) discrete_dir = 1;
			else discrete_dir = 2;

			// float dir = atan2f(dv, du);
			// unsigned char discrete_dir = static_cast<unsigned char>(
			// 													(dir + M_PI // all positive
			// 													+ M_PI_4f / 2.0f) // align regions
			// 													/ M_PI_4f); // 8 dirs
			if (!(dirs[(dv * width + du)] & 1 << discrete_dir)) continue;

			// normalize v displacement
			dun = (du - u) * iwidth;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}

		// Left side

		// u displacement
		du = u - rad;
		// normalized
		dun = -rad * iwidth;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (du < 0) du += width; 
		// set v limits
		start = max(v - rad + 1, 0);
		end = min(v + rad, height);

		// go through side
		for (int dv = start; dv < end; ++dv) {
			// check if local maxima in the given direction, skip if not

			int discrete_dir;
			if (dv - start <= rad / 2) discrete_dir = 3;
			else if (end - dv <= rad / 2) discrete_dir = 5;
			else discrete_dir = 4;

			// float dir = atan2f(dv, du);
			// unsigned char discrete_dir = static_cast<unsigned char>(
			// 													(dir + M_PI // all positive
			// 													+ M_PI_4f / 2.0f) // align regions
			// 													/ M_PI_4f); // 8 dirs
			if (!(dirs[(dv * width + du)] & 1 << discrete_dir)) continue;

			// normalize v displacement
			dvn = (dv - v) * iheight;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}


		// Bottom side

		// u displacement
		dv = v + rad;
		// normalized
		dvn = rad * iheight;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (dv >= height) dv -= height; 
		// set u limits
		start = max(u - rad + 1, 0);
		end = min(u + rad, width);

		// go through side
		for (int du = start; du < end; ++du) {
			// check if local maxima in the given direction, skip if not

			int discrete_dir;
			if (du - start <= rad / 2) discrete_dir = 5;
			else if (end - du <= rad / 2) discrete_dir = 7;
			else discrete_dir = 6;

			// float dir = atan2f(dv, du);
			// unsigned char discrete_dir = static_cast<unsigned char>(
			// 													(dir + M_PI // all positive
			// 													+ M_PI_4f / 2.0f) // align regions
			// 													/ M_PI_4f); // 8 dirs
			if (!(dirs[(dv * width + du)] & 1 << discrete_dir)) continue;

			// normalize v displacement
			dun = (du - u) * iwidth;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}
	}

	float ratio = sqrt(min_ratio2);
  // most of the data is on the low end...sqrting again spreads it better
  // (plus multiply is a cheap operation in shaders!)
  // -- Dummer
	ratio = sqrt(ratio);
	cone_map[(v * width + u) * 4 + 0] = heightmap[(v * width + u) * channels];
	cone_map[(v * width + u) * 4 + 1] = static_cast<unsigned char>(ratio * 255);
	cone_map[(v * width + u) * 4 + 2] = derivative_image[(v * width + u) * 3];
	cone_map[(v * width + u) * 4 + 3] = derivative_image[(v * width + u) * 3 + 1];
}

__global__ void create_cone_map_4dirs(unsigned char* heightmap, unsigned char* derivative_image, unsigned char* dirs, unsigned char* cone_map, int width, int height, int channels) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;

  if (u >= width || v >= height) return;

	float iwidth = 1.0f / width;
	float iheight = 1.0f / height;

	// TODO Why are we assuming run/rise = 1 exactly (instead of infinity)? Float textures?
	float min_ratio2 = 1.0f;

	// normalize height
	float h = heightmap[(v * width + u) * channels] / 255.0f;

	// init variables
	int du, dv;
	float dun, dvn;
	int start, end;

	// search in an increasing radius spiral around the texel
	for (int rad = 1;
			// TODO why the 1.1f?
			// otherwise we can't find anything steeper than the current min_ratio (see min_ratio assignment)
			rad * rad <= 1.1f * (1.0f - h) * width *
									 1.1f * (1.0f - h) * height *
									 min_ratio2 &&
			// because we started from 1, and further than (1.0f - h) * width, we couldn't find anything steeper than 1
			rad <= 1.1f * (1.0f - h) * width &&
			rad <= 1.1f * (1.0f - h) * height;
			++rad) {

		// Right side

		// u displacement
		du = u + rad;
		// normalized
		dun = rad * iwidth;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (du >= width) du -= width; 
		// set v limits
		start = max(v - rad, 0);
		end = min(v + rad - 1, height);

		// go through side
		for (int dv = start; dv < end; ++dv) {
			// check if local maxima in the given direction, skip if not

			if (!(dirs[(dv * width + du)] & 1 << 0)) continue;

			// normalize v displacement
			dvn = (dv - v) * iheight;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}

		// Top side

		// u displacement
		dv = v - rad;
		// normalized
		dvn = -rad * iheight;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (dv < 0) dv += height; 
		// set u limits
		start = max(u - rad, 0);
		end = min(u + rad - 1, width);

		// go through side
		for (int du = start; du < end; ++du) {
			// check if local maxima in the given direction, skip if not

			if (!(dirs[(dv * width + du)] & 1 << 2)) continue;

			// normalize v displacement
			dun = (du - u) * iwidth;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}

		// Left side

		// u displacement
		du = u - rad;
		// normalized
		dun = -rad * iwidth;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (du < 0) du += width; 
		// set v limits
		start = max(v - rad + 1, 0);
		end = min(v + rad, height);

		// go through side
		for (int dv = start; dv < end; ++dv) {
			// check if local maxima in the given direction, skip if not

			if (!(dirs[(dv * width + du)] & 1 << 4)) continue;

			// normalize v displacement
			dvn = (dv - v) * iheight;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}


		// Bottom side

		// u displacement
		dv = v + rad;
		// normalized
		dvn = rad * iheight;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (dv >= height) dv -= height; 
		// set u limits
		start = max(u - rad + 1, 0);
		end = min(u + rad, width);

		// go through side
		for (int du = start; du < end; ++du) {
			// check if local maxima in the given direction, skip if not

			if (!(dirs[(dv * width + du)] & 1 << 6)) continue;

			// normalize v displacement
			dun = (du - u) * iwidth;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}
	}

	float ratio = sqrt(min_ratio2);
  // most of the data is on the low end...sqrting again spreads it better
  // (plus multiply is a cheap operation in shaders!)
  // -- Dummer
	ratio = sqrt(ratio);
	cone_map[(v * width + u) * 4 + 0] = heightmap[(v * width + u) * channels];
	cone_map[(v * width + u) * 4 + 1] = static_cast<unsigned char>(ratio * 255);
	cone_map[(v * width + u) * 4 + 2] = derivative_image[(v * width + u) * 3];
	cone_map[(v * width + u) * 4 + 3] = derivative_image[(v * width + u) * 3 + 1];
}

__global__ void first_derivative(unsigned char* heightmap, unsigned char* fod_image, unsigned char* dirs, unsigned char* dirs_image, int* fod, int width, int height, int channels) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;

	 if (u >= width || v >= height) return;

	 // Possible optimizations:
	 //  - For every pixel, compute the sum with the side adjacent ones in both dimensions and save to shared memory
	//  - Step by channels instead of 1 in the loop (only worth it, if kernel lookup is independent)
	//  - Previtt operator instead of Sobel -> no kernel needed multiply by value/sign of index
	int hsum = 0;
	int vsum = 0;
	for (int dv = 0; dv < 3; ++dv) {
		for (int du = 0; du < 3; ++du) {
	     int cu = min(max(u + du - 1, 0), width - 1);
	     int cv = min(max(v + dv - 1, 0), height - 1);
	     int cidx = (cv * width + cu) * channels;
			hsum += heightmap[cidx] * hkernel[dv * 3 + du];
			vsum += heightmap[cidx] * vkernel[dv * 3 + du];
		}
	}

	// int grad[2] = {hsum, vsum};
	// int scale = std::sqrt(vsum * vsum + hsum * hsum);
	float dir = atan2f(vsum, hsum);
	unsigned char discrete_dir = static_cast<unsigned char>(
																(dir + M_PI // all positive
																+ M_PI_4f / 2.0f) // align regions
																/ M_PI_4f // 8 dirs
																) % 4; // the 4 we care about;

	int idx = v * width + u;

	fod_image[idx * 3 + 0] = (hsum + 1020) >> 3;
	fod_image[idx * 3 + 1] = (vsum + 1020) >> 3;
	fod_image[idx * 3 + 2] = 127;

	fod[idx * 2 + 0] = hsum;
	fod[idx * 2 + 1] = vsum;

	dirs[idx] = discrete_dir;

	dirs_image[idx] = dirs[idx] * 32 + 127;
}

__global__ void second_derivative(int* fod, unsigned char* sod_image, unsigned char* watershed_image, int width, int height) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;

  if (u >= width || v >= height) return;
	
  // Possible optimizations:
  //  - For every pixel, compute the sum with the side adjacent ones in both dimensions and save to shared memory
	//  - Step by channels instead of 1 in the loop (only worth it, if kernel lookup is independent)
	//  - Previtt operator instead of Sobel -> no kernel needed multiply by value/sign of index
	int hhsum = 0;
	int vhsum = 0;
	int hvsum = 0;
	int vvsum = 0;
	for (int dv = 0; dv < 3; ++dv) {
		for (int du = 0; du < 3; ++du) {
      int cu = min(max(u + du - 1, 0), width - 1);
      int cv = min(max(v + dv - 1, 0), height - 1);
      int cidx = (cv * width + cu) * 2;
			hhsum += fod[cidx + 0] * hkernel[dv * 3 + du];
			hvsum += fod[cidx + 1] * hkernel[dv * 3 + du];
			vhsum += fod[cidx + 0] * vkernel[dv * 3 + du];
			vvsum += fod[cidx + 1] * vkernel[dv * 3 + du];
		}
	}

	int idx = v * width + u;

	sod_image[idx * 4 + 0] = (hhsum + 1020) >> 3;
	sod_image[idx * 4 + 1] = (hvsum + 1020) >> 3;
	sod_image[idx * 4 + 2] = (vhsum + 1020) >> 3;
	sod_image[idx * 4 + 3] = (vvsum + 1020) >> 3;
	
	int val = hhsum * fod[idx * 2 + 1] * fod[idx * 2 + 1] - (hvsum + vhsum) * fod[idx * 2 + 0] * fod[idx * 2 + 1] + vvsum * fod[idx * 2 + 0] * fod[idx * 2 + 0];

	watershed_image[idx] = val < 0 ? 255 : 0;
}

__global__ void non_maximum_suppression(unsigned char* heightmap, unsigned char* dirs, unsigned char* watershed_image, unsigned char* suppressed_image, int width, int height, int channels) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;

  if (u >= width || v >= height) return;

	int du;
	int dv;

	switch (dirs[v * width + u]) {
		case 0:
			du = 0;
			dv = 1;
			break;
		case 1:
			du = -1;
			dv = 1;
			break;
		case 2:
			du = 1;
			dv = 0;
			break;
		case 3:
			du = 1;
			dv = 1;
			break;
	}

	if (watershed_image[v * width + u] == 0) {
		suppressed_image[v * width + u] = 0;
	} else
	if (u + du >= 0 && u + du < width &&
			v + dv >= 0 && v + dv < height &&
			heightmap[((v + dv) * width + u + du) * channels] > heightmap[(v * width + u) * channels]) {
			suppressed_image[v * width + u] = 0;
	} else
	if (u - du >= 0 && u - du < width &&
			v - dv >= 0 && v - dv < height &&
			heightmap[((v - dv) * width + u - du) * channels] > heightmap[(v * width + u) * channels]) {
			suppressed_image[v * width + u] = 0;
	} else {
		suppressed_image[v * width + u] = 255;
	}
}

__global__ void create_cone_map_analytic(unsigned char* heightmap, unsigned char* fod_image, unsigned char* dirs, unsigned char* suppressed_image, unsigned char* cone_map, int width, int height, int channels) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  int v = blockIdx.y * blockDim.y + threadIdx.y;

  if (u >= width || v >= height) return;

	float iwidth = 1.0f / width;
	float iheight = 1.0f / height;

	// TODO Why are we assuming run/rise = 1, instead of infinity? Float textures?
	float min_ratio2 = 1.0f;

	// normalize height
	float h = heightmap[(v * width + u) * channels] / 255.0f;

	// init variables
	int du, dv;
	float dun, dvn;
	int start, end;

	// search in an increasing radius spiral around the texel
	for (int rad = 1;
			// TODO why the 1.1f?
			// otherwise we can't find anything steeper than the current min_ratio (see min_ratio assignment)
			rad * rad <= 1.1f * (1.0f - h) * width *
									 1.1f * (1.0f - h) * height *
									 min_ratio2 &&
			// because we started from 1, and further than (1.0f - h) * width, we couldn't find anything steeper than 1
			rad <= 1.1f * (1.0f - h) * width &&
			rad <= 1.1f * (1.0f - h) * height;
			++rad) {

		// Right side

		// u displacement
		du = u + rad;
		// normalized
		dun = rad * iwidth;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (du >= width) du -= width; 
		// set v limits
		start = max(v - rad, 0);
		end = min(v + rad - 1, height);

		// go through side
		for (int dv = start; dv < end; ++dv) {
			// check if (suppressed) watershed point, skip if not
			if (suppressed_image[dv * width + du] == 0 || dirs[dv * width + du] == 0 || dirs[dv * width + du] == 2) continue;

			// normalize v displacement
			dvn = (dv - v) * iheight;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}

		// Top side

		// u displacement
		dv = v - rad;
		// normalized
		dvn = -rad * iheight;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (dv < 0) dv += height; 
		// set u limits
		start = max(u - rad, 0);
		end = min(u + rad - 1, width);

		// go through side
		for (int du = start; du < end; ++du) {
			// check if (suppressed) watershed point, skip if not
			if (suppressed_image[dv * width + du] == 0 || dirs[dv * width + du] == 1 || dirs[dv * width + du] == 3) continue;

			// normalize v displacement
			dun = (du - u) * iwidth;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}

		// Left side

		// u displacement
		du = u - rad;
		// normalized
		dun = -rad * iwidth;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (du < 0) du += width; 
		// set v limits
		start = max(v - rad + 1, 0);
		end = min(v + rad, height);

		// go through side
		for (int dv = start; dv < end; ++dv) {
			// check if (suppressed) watershed point, skip if not
			if (suppressed_image[dv * width + du] == 0 || dirs[dv * width + du] == 0 || dirs[dv * width + du] == 2) continue;

			// normalize v displacement
			dvn = (dv - v) * iheight;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}


		// Bottom side

		// u displacement
		dv = v + rad;
		// normalized
		dvn = rad * iheight;

		// TODO only if tileable option is set
		// loop around until reaching valid coordinates
		while (dv >= height) dv -= height; 
		// set u limits
		start = max(u - rad + 1, 0);
		end = min(u + rad, width);

		// go through side
		for (int du = start; du < end; ++du) {
			// check if (suppressed) watershed point, skip if not
			if (suppressed_image[dv * width + du] == 0 || dirs[dv * width + du] == 1 || dirs[dv * width + du] == 3) continue;

			// normalize v displacement
			dun = (du - u) * iwidth;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[(dv * width + du) * channels] / 255.0 - h;

			// if more steep than previous best, override
			if (dh > 0.0f && dh * dh * min_ratio2 > d2) {
				min_ratio2 = d2 / (dh * dh);
			}
		}
	}

	float ratio = sqrt(min_ratio2);
  // most of the data is on the low end...sqrting again spreads it better
  // (plus multiply is a cheap operation in shaders!)
  // -- Dummer
	ratio = sqrt(ratio);
	cone_map[(v * width + u) * 4 + 0] = heightmap[(v * width + u) * channels];
	cone_map[(v * width + u) * 4 + 1] = static_cast<unsigned char>(ratio * 255);
	cone_map[(v * width + u) * 4 + 2] = fod_image[(v * width + u) * 3];
	cone_map[(v * width + u) * 4 + 3] = fod_image[(v * width + u) * 3 + 1];
}
