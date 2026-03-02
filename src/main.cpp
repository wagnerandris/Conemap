// STD
#include <getopt.h>
#include <iostream>
#include <filesystem>
#include <vector>

#include "conemap.hpp"

int main(int argc, char* argv[]) {
	std::filesystem::path output_path = ".";
	std::vector<std::filesystem::path> heightmap_files;
	std::vector<std::filesystem::path> depthmap_files;
	bool analytic = false;

   // Define long options
	const struct option long_options[] = {
		{"help",     no_argument,       nullptr, 'h'},
		{"analytic", no_argument,       nullptr, 'a'},
		{"output",   required_argument, nullptr, 'o'},
		{"depthmap", required_argument, nullptr, 'd'},
		{nullptr, 0, nullptr, 0}
	};

	int opt;
	int option_index = 0;

	// Parse options
	while ((opt = getopt_long(argc, argv, "hao:d:", long_options, &option_index)) != -1) {
		switch (opt) {
		case 'h':
			std::cout << "Usage: " << argv[0] << " [-a] [-o OUTPUT] [-d DEPTH MAP...] HEIGHT MAP...\n"
				"Options:\n"
				"  -h, --help            \tproduce help message\n"
				"  -a, --analytic        \tanalytic generation mode\n"
				"  -o, --output PATH     \tpath to output folder (default: '.')\n"
				"  -d, --depthmap FILE...\tinput depth map files\n"
				"Positional arguments:\n"
				"  FILE... \tinput height map files";
			exit(0);
			break;

		case 'a':
			analytic = true;
			break;

		case 'o':
			output_path = optarg;
			break;

		case 'd':
			depthmap_files.emplace_back(optarg);

			// Collect additional non-option arguments following -d
			while (optind < argc && argv[optind][0] != '-') {
				depthmap_files.emplace_back(argv[optind++]);
			}
			break;

		default:
			// Unknown option or missing argument
			return 1;
		}
	}

	// Remaining positional arguments
	while (optind < argc) {
		heightmap_files.emplace_back(argv[optind++]);
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
		if (analytic) conemap::analytic(output_path, file);
		else conemap::discrete(output_path, file);

	}

	// OK
	for (auto file : depthmap_files) {
		if (!std::filesystem::exists(file) || !std::filesystem::is_regular_file(file)) {
			std::cerr << "Error: " << file << " is not a file.\n";
			continue;
		}
		if (analytic) conemap::analytic(output_path, file, true);
		else conemap::discrete(output_path, file, true);
	}

	return 0;
}
