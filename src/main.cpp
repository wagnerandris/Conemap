// STD
#include <iostream>
#include <filesystem>
#include <vector>

// 3rd party
#include <cxxopts.hpp>

#include "conemap.hpp"

int main(int argc, char* argv[]) {
    std::filesystem::path output_path = ".";
    std::vector<std::filesystem::path> heightmap_files;
    std::vector<std::filesystem::path> depthmap_files;
    bool analytic = false;
    bool measure = false;

    try {
        cxxopts::Options options(argv[0], "Cone map generator");

        options.add_options()
            ("h,help", "produce help message")
            ("m,measure", "measure performance for different generation methods")
            ("a,analytic", "analytic generation mode")
            ("o,output", "output folder", cxxopts::value<std::string>()->default_value("."))
            ("d,depthmap", "input depth map files", cxxopts::value<std::vector<std::string>>())
            ("heightmaps", "input height map files", cxxopts::value<std::vector<std::string>>());

        options.parse_positional({"heightmaps"});
        options.positional_help("HEIGHT MAP...");

        auto result = options.parse(argc, argv);

        // Help
        if (result.count("help")) {
            std::cout << options.help() << "\n";
            return 0;
        }

        // Flags
        measure = result.count("measure") > 0;
        analytic = result.count("analytic") > 0;

        // Output path
        output_path = result["output"].as<std::string>();

        // Depthmaps
        if (result.count("depthmap")) {
            for (const auto& f : result["depthmap"].as<std::vector<std::string>>()) {
                depthmap_files.emplace_back(f);
            }
        }

        // Heightmaps (positional)
        if (result.count("heightmaps")) {
            for (const auto& f : result["heightmaps"].as<std::vector<std::string>>()) {
                heightmap_files.emplace_back(f);
            }
        }

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing options: " << e.what() << "\n";
        return 1;
    }

    // No input
    if (heightmap_files.empty() && depthmap_files.empty()) {
        std::cerr << "Error: No input files provided.\n";
        return 1;
    }

    // Output dir handling
    if (!std::filesystem::exists(output_path)) {
        std::filesystem::create_directory(output_path);
    } else if (!std::filesystem::is_directory(output_path)) {
        std::cerr << "Error: " << output_path << " is not a directory.\n";
        return 1;
    }

    // Heightmaps
    for (const auto& file : heightmap_files) {
        if (!std::filesystem::exists(file) || !std::filesystem::is_regular_file(file)) {
            std::cerr << "Error: " << file << " is not a file.\n";
            continue;
        }

        if (measure) conemap::measure(output_path, file, false);
        else if (analytic) conemap::analytic(output_path, file, false);
        else conemap::discrete(output_path, file, false);
    }

    // Depthmaps
    for (const auto& file : depthmap_files) {
        if (!std::filesystem::exists(file) || !std::filesystem::is_regular_file(file)) {
            std::cerr << "Error: " << file << " is not a file.\n";
            continue;
        }

        if (measure) conemap::measure(output_path, file, true);
        else if (analytic) conemap::analytic(output_path, file, true);
        else conemap::discrete(output_path, file, true);
    }

    return 0;
}