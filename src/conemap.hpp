#pragma once

#include <filesystem>

namespace conemap {

void measure(const std::filesystem::path output_path, const std::filesystem::path filepath, const bool depthmap);

std::filesystem::path analytic(const std::filesystem::path output_path, const std::filesystem::path filepath, const bool depthmap);

std::filesystem::path discrete(const std::filesystem::path output_path, const std::filesystem::path filepath, const bool depthmap);

}
