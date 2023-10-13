#pragma once

#include <string>

namespace megamol::power {
void DataverseWriter(std::string const& dataverse_path, std::string const& doi, std::string const& filepath, char const* key);
}
