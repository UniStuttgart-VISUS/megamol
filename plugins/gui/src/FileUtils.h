/*
 * FileUtils.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_FILEUTILS_INCLUDED
#define MEGAMOL_GUI_FILEUTILS_INCLUDED


#if defined(_HAS_CXX17) || ((defined(_MSC_VER) && (_MSC_VER > 1916))) // C++2017 or since VS2019
#    include <filesystem>
namespace fsns = std::filesystem;
#else
// WINDOWS
#    ifdef _WIN32
#        include <filesystem>
namespace fsns = std::experimental::filesystem;
#    else
// LINUX
#        include <experimental/filesystem>
namespace fsns = std::experimental::filesystem;
#    endif
#endif

#include "mmcore/AbstractNamedObject.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/Module.h"

#include "vislib/sys/AbstractReaderWriterLock.h"

#include <fstream>
#include <iostream>
#include <string>


namespace megamol {
namespace gui {


/**
 * Check if given file or directory exists.
 *
 * @param path  The file or directory path.
 */
template <typename T> bool PathExists(const T& path_str) {
    auto path = static_cast<fsns::path>(path_str);
    return fsns::exists(path);
}


/**
 * Check if file exists and has specified file extension.
 *
 * @param path  The file or directory path.
 * @param ext   The extension the given file should have.
 */
template <typename T> bool HasExistingFileExtension(const T& path_str, const std::string& ext) {
    auto path = static_cast<fsns::path>(path_str);
    if (!fsns::exists(path)) {
        return false;
    }
    return (path.extension().string() == ext);
}


/**
 * Check if file has specified file extension.
 *
 * @param path  The file or directory path.
 * @param ext   The extension the given file should have.
 */
template <typename T> bool HasFileExtension(const T& path_str, const std::string& ext) {
    auto path = static_cast<fsns::path>(path_str);
    return (path.extension().string() == ext);
}


/**
 * Search recursively for file or path beginning at given directory.
 *
 * @param file          The file to search for.
 * @param searchPath    The path of a directory as start for recursive search.
 *
 * @return              The complete path of the found file, empty string otherwise.
 */
template <typename T, typename S> std::string SearchFileRecursive(const T& search_path_str, const S& search_file_str) {
    auto search_path = static_cast<fsns::path>(search_path_str);
    auto file_path = static_cast<fsns::path>(search_file_str);
    std::string found_path;
    for (const auto& entry : fsns::recursive_directory_iterator(search_path)) {
        if (entry.path().filename() == file_path) {
            found_path = entry.path().string();
            break;
        }
    }
    return found_path;
}


/**
 * Save currently loaded project to lua file.
 *
 * @param projectFilename The file name for the project.
 * @param coreInstance    The pointer to the core instance.
 *
 * @return True on success, false otherwise.
 */
inline bool SaveProjectFile(const std::string& project_filename, megamol::core::CoreInstance* core_instance) {

    if (core_instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to CoreInstance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    std::string serInstances, serModules, serCalls, serParams;
    core_instance->SerializeGraph(serInstances, serModules, serCalls, serParams);
    auto confstr = serInstances + "\n" + serModules + "\n" + serCalls + "\n" + serParams + "\n";

    try {
        std::ofstream file;
        file.open(project_filename);
        if (file.good()) {
            file << confstr.c_str();
            file.close();
        } else {
            vislib::sys::Log::DefaultLog.WriteError(
                "Unable to create project file. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            file.close();
            return false;
        }
    } catch (std::exception e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("Unknown Error. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_FILEUTILS_INCLUDED