#include "kernels.cuh"
#include <cmath>


/* Utils */

__global__ void invert(unsigned char* data, int width, int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;

	data[v * width + u] = 255 - data[v * width + u];
}

__global__ void bits_to_image(unsigned char* data, unsigned char* output_image, int width, int height, unsigned char bitmask) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;

	output_image[v * width + u] = data[v * width + u] & bitmask ? 255 : 0;
}


/* Derivatives */

__global__ void fod(unsigned char* heightmap, int* fods, float* fod_dirs, int width, int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;
	
	int idx = v * width + u;

	int hsum = 0;
	int vsum = 0;
	for (int dv = 0; dv < 3; ++dv) {
		for (int du = 0; du < 3; ++du) {
			int cu = min(max(u + du - 1, 0), width - 1);
			int cv = min(max(v + dv - 1, 0), height - 1);
			int cidx = (cv * width + cu);
			hsum += heightmap[cidx] * hkernel[dv * 3 + du];
			vsum += heightmap[cidx] * vkernel[dv * 3 + du];
		}
	}

	fod_dirs[idx] = atan2f(vsum, hsum);

	fods[idx * 2 + 0] = hsum;
	fods[idx * 2 + 1] = vsum;
}


__global__ void watershed(int* fods, bool* watersheds, int width, int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;
	
	// Second order derivative
	int hhsum = 0;
	int vhsum = 0;
	int hvsum = 0;
	int vvsum = 0;
	for (int dv = 0; dv < 3; ++dv) {
		for (int du = 0; du < 3; ++du) {
			int cu = min(max(u + du - 1, 0), width - 1);
			int cv = min(max(v + dv - 1, 0), height - 1);
			int cidx = (cv * width + cu) * 2;
			hhsum += fods[cidx + 0] * hkernel[dv * 3 + du];
			hvsum += fods[cidx + 1] * hkernel[dv * 3 + du];
			vhsum += fods[cidx + 0] * vkernel[dv * 3 + du];
			vvsum += fods[cidx + 1] * vkernel[dv * 3 + du];
		}
	}

	int idx = v * width + u;
	
	watersheds[idx] =
		hhsum * fods[idx * 2 + 1] * fods[idx * 2 + 1] -
		(hvsum + vhsum) * fods[idx * 2 + 0] * fods[idx * 2 + 1] +
		vvsum * fods[idx * 2 + 0] * fods[idx * 2 + 0]
		< 0;
}


/* Local maxima */

__global__ void non_maximum_suppression(unsigned char* heightmap, float* fod_dirs, bool* watershed, bool* suppressed, int width, int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;
	
	int idx = v * width + u;

	int du;
	int dv;

	if (!watershed[idx]) {
		suppressed[idx] = 0;
		return;
	}

	// get the nearest of the directions of neighbouring texels to the gradient direction
	unsigned char discrete_dir = static_cast<unsigned char>(
																(fod_dirs[idx] + M_PI // all positive
																+ M_PI_4f / 2.0f) // align regions
																/ M_PI_4f // 8 dirs
																) % 4; // the 4 we care about;

	// check in the direction orthogonal to the gradient
	switch (discrete_dir) {
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

	// if one of the neighbours has greater height, suppress the current texel
	if (u + du >= 0 && u + du < width &&
			v + dv >= 0 && v + dv < height &&
			heightmap[((v + dv) * width + u + du)] > heightmap[idx]) {
			suppressed[idx] = 0;
	} else
	if (u - du >= 0 && u - du < width &&
			v - dv >= 0 && v - dv < height &&
			heightmap[((v - dv) * width + u - du)] > heightmap[idx]) {
			suppressed[idx] = 0;
	} else {
		suppressed[idx] = watershed[idx];
	}
}

__global__ void local_max_8dir(unsigned char* heightmap, unsigned char* local_max_8dirs, int width, int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;

	int h = heightmap[v * width + u];

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
				int hn = u_good ? heightmap[vn * width + un] : -1;
				int hp = v_good ? heightmap[vp * width + up] : -1;

				// local max in hp to hn direction (if there is a plateau, we need its last point)
				if (h > hn && h >= hp) {
						result |= (1 << dir);
				}
		}

		local_max_8dirs[v * width + u] = result;
}


/* Cone maps */

__global__ void create_cone_map_analytic(unsigned char* heightmap, bool* suppressed, float* fod_dirs, int* fods, unsigned char* cone_map, int width, int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;

	int idx = v * width + u;

	float iwidth = 1.0f / width;
	float iheight = 1.0f / height;

	// TODO Why are we assuming run/rise = 1, instead of infinity? Float textures?
	float min_ratio2 = 1.0f;

	// normalize height
	float h = heightmap[idx] / 255.0f;

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
			if (!suppressed[dv * width + du] ||
					abs(remainderf((fod_dirs[dv * width + du] - atan2f(dv, du) - M_PI_2), M_PI)) > (M_PI / 8.0))
				continue;

			// normalize v displacement
			dvn = (dv - v) * iheight;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[dv * width + du] / 255.0 - h;

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
			if (!suppressed[dv * width + du] ||
					abs(remainderf((fod_dirs[dv * width + du] - atan2f(dv, du) - M_PI_2), M_PI)) > (M_PI / 8.0))
				continue;

			// normalize v displacement
			dun = (du - u) * iwidth;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[dv * width + du] / 255.0 - h;

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
			if (!suppressed[dv * width + du] ||
					abs(remainderf((fod_dirs[dv * width + du] - atan2f(dv, du) - M_PI_2), M_PI)) > (M_PI / 8.0))
				continue;

			// normalize v displacement
			dvn = (dv - v) * iheight;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[dv * width + du] / 255.0 - h;

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
			if (!suppressed[dv * width + du] ||
					abs(remainderf((fod_dirs[dv * width + du] - atan2f(dv, du) - M_PI_2), M_PI)) > (M_PI / 8.0))
				continue;

			// normalize v displacement
			dun = (du - u) * iwidth;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[dv * width + du] / 255.0 - h;

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
	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<unsigned char>(ratio * 255);
	cone_map[idx * 4 + 2] = (fods[idx * 2] + 1020) / 8;
	cone_map[idx * 4 + 3] = (fods[idx * 2 + 1] + 1020) / 8;
}


__global__ void create_cone_map_8dir(unsigned char* heightmap, unsigned char* local_max_8dirs, unsigned char* cone_map, int width, int height) {
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	if (u >= width || v >= height) return;

/* First order derivative */
	int hsum = 0;
	int vsum = 0;
	for (int dv = 0; dv < 3; ++dv) {
		for (int du = 0; du < 3; ++du) {
			 int cu = min(max(u + du - 1, 0), width - 1);
			 int cv = min(max(v + dv - 1, 0), height - 1);
			 int cidx = (cv * width + cu);
			hsum += heightmap[cidx] * hkernel[dv * 3 + du];
			vsum += heightmap[cidx] * vkernel[dv * 3 + du];
		}
	}

	unsigned char dhdu = (hsum + 1020) / 8;
  unsigned char dhdv = (vsum + 1020) / 8;

/*Cone ratios*/
	int idx = v * width + u;

	float iwidth = 1.0f / width;
	float iheight = 1.0f / height;

	// TODO Why are we assuming run/rise = 1 exactly (instead of infinity)? Float textures?
	float min_ratio2 = 1.0f;

	// normalize height
	float h = heightmap[idx] / 255.0f;

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
			if (!(local_max_8dirs[dv * width + du] & 1 << discrete_dir)) continue;

			// normalize v displacement
			dvn = (dv - v) * iheight;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[dv * width + du] / 255.0 - h;

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
			if (!(local_max_8dirs[dv * width + du] & 1 << discrete_dir)) continue;

			// normalize v displacement
			dun = (du - u) * iwidth;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[dv * width + du] / 255.0 - h;

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
			if (!(local_max_8dirs[dv * width + du] & 1 << discrete_dir)) continue;

			// normalize v displacement
			dvn = (dv - v) * iheight;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[dv * width + du] / 255.0 - h;

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
			if (!(local_max_8dirs[dv * width + du] & 1 << discrete_dir)) continue;

			// normalize v displacement
			dun = (du - u) * iwidth;

			// distance squared
			float d2 = dun * dun + dvn * dvn;

			// height difference
			float dh = heightmap[dv * width + du] / 255.0 - h;

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
	cone_map[idx * 4 + 0] = heightmap[idx];
	cone_map[idx * 4 + 1] = static_cast<unsigned char>(ratio * 255);
	cone_map[idx * 4 + 2] = dhdu;
	cone_map[idx * 4 + 3] = dhdv;
}
