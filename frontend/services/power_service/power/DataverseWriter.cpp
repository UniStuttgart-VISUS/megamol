#include "DataverseWriter.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include <dataverse/dataverse_connection.h>

namespace megamol::power {
void DataverseWriter(std::string const& dataverse_path, std::string const& doi, std::string const& filepath,
    char const* key, char& signal) {
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
                }, &signal);
    } catch (...) {
        signal = false;
    }
    //std::thread t(
    //    [](std::string const& dataverse_path, std::string const& doi, std::string const& filepath, char const* key) {
    //        std::atomic_bool running(true);
    //        dataverse_connection dataverse;
    //        dataverse.base_path(make_const_narrow_string(dataverse_path, CP_OEMCP))
    //            .api_key(make_const_narrow_string(key, CP_OEMCP))
    //            .upload(
    //                make_const_narrow_string(doi, CP_OEMCP), make_const_narrow_string(filepath, CP_OEMCP),
    //                /*make_const_narrow_string("Test Data", CP_OEMCP), make_const_narrow_string("", CP_OEMCP), nullptr, 0,
    //                false,*/
    //                [](const blob& result, void* context) {
    //                    std::string output(result.as<char>(), result.size());
    //                    std::cout << convert<char>(convert<wchar_t>(output, CP_UTF8), CP_OEMCP) << std::endl;
    //                    std::cout << std::endl;
    //                    (*static_cast<decltype(running)*>(context)).store(false);
    //                },
    //                [](const int error, const char* msg, const char* cat, narrow_string::code_page_type cp,
    //                    void* context) {
    //                    std::cerr << msg << std::endl << std::endl;
    //                    (*static_cast<decltype(running)*>(context)).store(false);
    //                },
    //                &running);

    //        while (running.load()) {
    //            std::this_thread::sleep_for(std::chrono::milliseconds(100));
    //        }
    //    },
    //    dataverse_path, doi, filepath, key);
    //t.detach();
}
} // namespace megamol::power
