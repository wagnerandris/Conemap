#pragma once

#include <filesystem>

namespace conemap {

std::filesystem::path analytic(std::filesystem::path output_path, std::string filepath, bool depthmap = false);

std::filesystem::path discrete(std::filesystem::path output_path, std::filesystem::path filepath, bool depthmap = false);

}
