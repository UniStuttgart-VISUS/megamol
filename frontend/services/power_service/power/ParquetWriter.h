#pragma once

#ifdef MEGAMOL_USE_POWER

#include <filesystem>
#include <vector>

#include "MetaData.h"
#include "SampleBuffer.h"
#include "Utility.h"

namespace megamol::power {
/**
 * @brief Writer for @c value_map_t container to parquet file.
 * @param file_path The filepath.
 * @param values_map The data to write.
 * @param meta The MetaData object to embed in the output file.
 */
void ParquetWriter(
    std::filesystem::path const& file_path, value_map_t const& values_map, MetaData const* meta = nullptr);

/**
 * @brief Writer a set of @c SampleBuffer to parquet file.
 * @param file_path The filepath.
 * @param buffers The data to write.
 * @param meta The MetaData object to embed in the output file.
 */
void ParquetWriter(
    std::filesystem::path const& file_path, std::vector<SampleBuffer> const& buffers, MetaData const* meta = nullptr);
} // namespace megamol::power

#endif
