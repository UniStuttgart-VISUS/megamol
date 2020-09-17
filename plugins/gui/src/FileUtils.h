/*
 * FileUtils.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_FILEUTILS_INCLUDED
#define MEGAMOL_GUI_FILEUTILS_INCLUDED

/// There is a CMake exeption for the cluster "stampede2" running CentOS, which undefines GUI_USE_FILESYSTEM.
#ifdef GUI_USE_FILESYSTEM
#    if defined(_HAS_CXX17) || ((defined(_MSC_VER) && (_MSC_VER > 1916))) // C++2017 or since VS2019
#        include <filesystem>
namespace stdfs = std::filesystem;
#    else
// WINDOWS
#        ifdef _WIN32
#            include <filesystem>
namespace stdfs = std::experimental::filesystem;
#        else
// LINUX
#            include <experimental/filesystem>
namespace stdfs = std::experimental::filesystem;
#        endif
#    endif
#endif // GUI_USE_FILESYSTEM

#include "GUIUtils.h"

#include <fstream>
#include <iostream>

#include "vislib/sys/FastFile.h"


namespace megamol {
namespace gui {

/**
 * File utility functions.
 */
class FileUtils {
public:
    /**
     * Load raw data from file (e.g. texture data)
     */
    static size_t LoadRawFile(std::string name, void** outData);

    /**
     * Check if file exists and has specified file extension.
     *
     * @param path  The file or directory path.
     * @param ext   The extension the given file should have.
     */
    template <typename T> static bool FilesExistingExtension(const T& path_str, const std::string& ext);

    /**
     * Check if file has specified file extension.
     *
     * @param path  The file or directory path.
     * @param ext   The extension the given file should have.
     */
    template <typename T> static bool FileExtension(const T& path_str, const std::string& ext);

    /**
     * Search recursively for file or path beginning at given directory.
     *
     * @param file          The file to search for.
     * @param searchPath    The path of a directory as start for recursive search.
     *
     * @return              The complete path of the found file, empty string otherwise.
     */
    template <typename T, typename S>
    static std::string SearchFileRecursive(const T& search_path_str, const S& search_file_str);

    /**
     * Writes content to file.
     *
     * @param filename      The file name of the file.
     * @param in_content    The content to wirte to the file.
     *
     * @return True on success, false otherwise.
     */
    static bool WriteFile(const std::string& filename, const std::string& in_content);

    /**
     * Read content from file.
     *
     * @param filename      The file name of the file.
     * @param in_content    The content to wirte to the file.
     *
     * @return True on success, false otherwise.
     */
    static bool ReadFile(const std::string& filename, std::string& out_content);

private:
    FileUtils(void);
    ~FileUtils(void) = default;
};


template <typename T> bool megamol::gui::FileUtils::FilesExistingExtension(const T& path_str, const std::string& ext) {
#ifdef GUI_USE_FILESYSTEM
    auto path = static_cast<stdfs::path>(path_str);
    if (!stdfs::exists(path) || !stdfs::is_regular_file(path)) {
        return false;
    }
    return (path.extension().generic_u8string() == ext);
#else
    return false;
#endif // GUI_USE_FILESYSTEM
}


template <typename T> bool megamol::gui::FileUtils::FileExtension(const T& path_str, const std::string& ext) {
#ifdef GUI_USE_FILESYSTEM
    auto path = static_cast<stdfs::path>(path_str);
    return (path.extension().generic_u8string() == ext);
#else
    return false;
#endif // GUI_USE_FILESYSTEM
}


template <typename T, typename S>
std::string megamol::gui::FileUtils::SearchFileRecursive(const T& search_path_str, const S& search_file_str) {
#ifdef GUI_USE_FILESYSTEM
    auto search_path = static_cast<stdfs::path>(search_path_str);
    auto file_path = static_cast<stdfs::path>(search_file_str);
    std::string found_path;
    for (const auto& entry : stdfs::recursive_directory_iterator(search_path)) {
        if (entry.path().filename() == file_path) {
            found_path = entry.path().generic_u8string();
            break;
        }
    }
    return found_path;
#else
    return std::string();
#endif // GUI_USE_FILESYSTEM
}


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_FILEUTILS_INCLUDED
