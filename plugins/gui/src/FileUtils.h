/*
 * FileUtils.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#if _HAS_CXX17
#    include <filesystem> // directory_iterator
namespace fsns = std::filesystem;
#else
// WINDOWS
#    ifdef _WIN32
#        include <filesystem>
#    else
// LINUX
#        include <experimental/filesystem>
#    endif
namespace fsns = std::experimental::filesystem;
#endif

#include <fstream>
#include <iostream>
#include <string>

#include "mmcore/AbstractNamedObject.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/Module.h"

#include "vislib/sys/AbstractReaderWriterLock.h"


namespace megamol {
namespace gui {

/** Type for filesystem paths. */
typedef fsns::path PathType;

/**
 * Check if given file or directory exists.
 *
 * @param path  The file or directory path.
 */
inline bool PathExists(PathType path) { return fsns::exists(path); }

/**
 * Check if file exists and has specified file extension.
 *
 * @param path  The file or directory path.
 * @param ext   The extension the given file should have.
 */
inline bool HasExistingFileExtension(PathType path, std::string ext) {
    if (!fsns::exists(static_cast<PathType>(path))) {
        return false;
    }
    return (path.extension().generic_string() == ext);
}

/**
 * Check if file has specified file extension.
 *
 * @param path  The file or directory path.
 * @param ext   The extension the given file should have.
 */
inline bool HasFileExtension(PathType path, std::string ext) { return (path.extension().generic_string() == ext); }

/**
 * Search recursively for file or path beginning at given directory.
 *
 * @param file          The file to search for.
 * @param searchPath    The path of a directory as start for recursive search.
 *
 * @return              The complete path of the found file, empty string otherwise.
 */
inline std::string SearchFileRecursive(std::string file, PathType searchPath) {
    std::string foundPath;
    for (const auto& entry : fsns::recursive_directory_iterator(searchPath)) {
        if (entry.path().filename().generic_string() == file) {
            foundPath = entry.path().generic_string();
            break;
        }
    }
    return foundPath;
}


/**
 * Save project to file.
 *
 * @param projectFilename The file name for the project.
 * @param coreInstance    The pointer to the core instance.
 *
 * @return True on success, false otherwise.
 */
inline bool SaveProjectFile(std::string projectFilename, megamol::core::CoreInstance* coreInstance) {

    if (coreInstance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[SaveProjectFile] Pointer to CoreInstance is nullptr.");
        return false;
    }
    std::string serInstances, serModules, serCalls, serParams;
    coreInstance->SerializeGraph(serInstances, serModules, serCalls, serParams);
    auto confstr = serInstances + "\n" + serModules + "\n" + serCalls + "\n" + serParams + "\n";

    try {
        std::ofstream file;
        file.open(projectFilename);
        if (file.good()) {
            file << confstr.c_str();
            file.close();
        } else {
            vislib::sys::Log::DefaultLog.WriteError("[SaveProjectFile] Couldn't create project file.");
            file.close();
            return false;
        }
    } catch (...) {
    }

    return true;
}


} // namespace gui
} // namespace megamol
