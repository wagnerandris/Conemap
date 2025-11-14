// STD
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

// Boost
#include <boost/program_options.hpp>

#include "conemap.hpp"

int main(int argc, char* argv[]) {
	std::filesystem::path output_path;
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
		conemap::analytic(output_path, file.c_str());
		conemap::discrete(output_path, file);

	}

	// OK
	for (auto file : depthmap_files) {
		if (!std::filesystem::exists(file) || !std::filesystem::is_regular_file(file)) {
			std::cerr << "Error: " << file << " is not a file.\n";
			continue;
		}
		conemap::analytic(output_path, file, true);
		conemap::discrete(output_path, file, true);
	}

	return 0;
}
