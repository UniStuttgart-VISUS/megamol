#pragma once

#ifdef MEGAMOL_USE_POWER

#include <filesystem>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "power/MetaData.h"
#include "power/SampleBuffer.h"

namespace megamol::power {
void ParquetWriter(std::filesystem::path const& file_path,
    std::unordered_map<std::string, std::variant<std::vector<float>, std::vector<int64_t>>> const& values_map,
    MetaData const* meta = nullptr);
void ParquetWriter(
    std::filesystem::path const& file_path, std::vector<SampleBuffer> const& buffers, MetaData const* meta = nullptr);
} // namespace megamol::power

#endif
