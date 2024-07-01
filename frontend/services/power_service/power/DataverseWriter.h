#pragma once

#ifdef MEGAMOL_USE_POWER

#include <string>

namespace megamol::power {
/**
 * @brief Writes the specified file into the Dataverse.
 * @param dataverse_path API path to the Dataverse.
 * @param doi DOI of the dataset to write into in the Dataverse.
 * @param filepath The path to the file to be written into the Dataverse.
 * @param key API token of the Dataverse.
 * @param signal True while the file is being written.
 */
void DataverseWriter(std::string const& dataverse_path, std::string const& doi, std::string const& filepath,
    char const* key, char& signal);
} // namespace megamol::power

#endif
