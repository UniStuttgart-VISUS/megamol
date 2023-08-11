#pragma once

#ifdef MEGAMOL_USE_POWER

#include <filesystem>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace megamol::frontend {
void ParquetWriter(std::filesystem::path const& file_path,
    std::unordered_map<std::string, std::variant<std::vector<float>, std::vector<int64_t>>> const& values_map);
} // namespace megamol::frontend

#endif
