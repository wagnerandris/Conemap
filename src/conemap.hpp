#pragma once

#include <filesystem>

void analytic(std::filesystem::path output_path, std::string filepath, bool depthmap = false);

void discrete(std::filesystem::path output_path, std::filesystem::path filepath, bool depthmap = false);
