#pragma once

#ifdef MEGAMOL_USE_POWER

#include <filesystem>
#include <unordered_map>
#include <vector>

namespace megamol::frontend {
void ParquetWriter(std::filesystem::path const& file_path,
    std::unordered_map<std::string, std::vector<float>> const& values_map);
} // namespace megamol::frontend

#endif
