#include "DataverseWriter.h"

#ifdef MEGAMOL_USE_POWER

#include <iostream>
#include <string>

#include <dataverse/dataverse_connection.h>

namespace megamol::power {
void DataverseWriter(std::string const& dataverse_path, std::string const& doi, std::string const& filepath,
    char const* key, char& signal) {
#if WIN32
    using namespace visus::dataverse;
    signal = true;
    try {
        dataverse_connection dataverse;
        dataverse.base_path(make_const_narrow_string(dataverse_path, CP_OEMCP))
            .api_key(make_const_narrow_string(key, CP_OEMCP))
            .upload(
                make_const_narrow_string(doi, CP_OEMCP), make_const_narrow_string(filepath, CP_OEMCP),
                /*make_const_narrow_string("Test Data", CP_OEMCP), make_const_narrow_string("", CP_OEMCP), nullptr, 0,
                false,*/
                [](const blob& result, void* context) {
                    std::string output(result.as<char>(), result.size());
                    std::cout << convert<char>(convert<wchar_t>(output, CP_UTF8), CP_OEMCP) << std::endl;
                    std::cout << std::endl;
                    *static_cast<char*>(context) = false;
                },
                [](const int error, const char* msg, const char* cat, narrow_string::code_page_type cp, void* context) {
                    std::cerr << msg << std::endl << std::endl;
                    *static_cast<char*>(context) = false;
                },
                &signal);
    } catch (...) {
        signal = false;
    }
#endif
}
} // namespace megamol::power

#endif
