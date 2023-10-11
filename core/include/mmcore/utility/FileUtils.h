/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <codecvt>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <istream>
#include <locale>
#include <string>
#include <vector>

#include "mmcore/utility/log/Log.h"
#include "vislib/UTF8Encoder.h"

namespace megamol::core::utility {

// #### Utility string conversion functions ############################ //

static inline std::string WChar2Utf8String(const std::wstring& wstr) {
    return std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(wstr);
}

// ##################################################################### //
/**
 * File utility functions.
 */
class FileUtils {
public:
    /**
     * Check if file exists.
     *
     * @param path_str  The file or directory path.
     */
    template<typename T>
    static bool FileExists(const T& path_str);

    /**
     * Check if any file exists and has specified file extension.
     *
     * @param path_str  The file or directory path.
     * @param ext   The extension the given file should have.
     */
    template<typename T>
    static bool FileWithExtensionExists(const T& path_str, const std::string& ext);

    /**
     * Check if any file exists and has specified file extension.
     *
     * @param path_str  The file or directory path.
     * @param ext   The extension the given file should have.
     */
    template<typename T>
    static bool FileHasExtension(const T& path_str, const std::string& ext);

    /**
     * Get stem of file name (file name __without__ leading path and without extension).
     *
     * @param path_str  The file or directory path.
     */
    template<typename T>
    static std::string GetFileNameStem(const T& path_str);

    /**
     * Get stem of file path (file name __with__ leading path but without extension).
     *
     * @param path_str  The file or directory path.
     */
    template<typename T>
    static std::string GetFilePathStem(const T& path_str);

    /**
     * Search recursively for file or path beginning at given directory.
     *
     * @param file          The file to search for.
     * @param searchPath    The path of a directory as start for recursive search.
     *
     * @return              The complete path of the found file, empty string otherwise.
     */
    template<typename T, typename S>
    static std::string SearchFileRecursive(const T& search_path_str, const S& search_file_str);

    /**
     * Writes content to file.
     *
     * @param filename      The file name of the file.
     * @param in_content    The content to wirte to the file.
     * @param silent        Disable log output.
     *
     * @return True on success, false otherwise.
     */
    static bool WriteFile(const std::filesystem::path& filename, const std::string& in_content, bool silent = false);

    /**
     * Read content from file.
     *
     * @param filename      The file name of the file.
     * @param out_content   The content to read from file.
     * @param silent        Disable log output.
     *
     * @return True on success, false otherwise.
     */
    static bool ReadFile(const std::filesystem::path& filename, std::string& out_content, bool silent = false);

    /**
     * Load raw data from file (e.g. texture data)
     */
    static bool LoadRawFile(const std::filesystem::path& filename, std::vector<char>& out_data);

private:
    FileUtils() = default;
    ~FileUtils() = default;
};


template<typename T>
bool megamol::core::utility::FileUtils::FileExists(const T& path_str) {
    auto filepath = std::filesystem::u8path(path_str);
    try {
        if (std::filesystem::exists(filepath) && std::filesystem::is_regular_file(filepath)) {
            return true;
        }
    } catch (std::filesystem::filesystem_error const& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
    }
    return false;
}


template<typename T>
bool megamol::core::utility::FileUtils::FileWithExtensionExists(const T& path_str, const std::string& ext) {
    try {
        if (FileUtils::FileExists<T>(path_str)) {
            auto filepath = std::filesystem::u8path(path_str);
            return (filepath.extension().generic_string() == std::string("." + ext));
        }
    } catch (std::filesystem::filesystem_error const& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
    }
    return false;
}


template<typename T>
bool megamol::core::utility::FileUtils::FileHasExtension(const T& path_str, const std::string& ext) {
    auto filepath = std::filesystem::u8path(path_str);
    return (filepath.extension().generic_string() == ext);
}


template<typename T>
std::string megamol::core::utility::FileUtils::GetFileNameStem(const T& path_str) {
    try {
        auto filepath = std::filesystem::u8path(path_str);
        std::string filename;
        if (filepath.has_stem()) {
            filename = filepath.stem().generic_string();
        }
        return filename;
    } catch (std::filesystem::filesystem_error const& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return std::string();
    }
}


template<typename T>
std::string megamol::core::utility::FileUtils::GetFilePathStem(const T& path_str) {
    try {
        auto filepath = std::filesystem::u8path(path_str);
        std::string filename;
        if (filepath.has_stem()) {
            if (filepath.has_parent_path()) {
                filename = (filepath.parent_path() / filepath.stem()).generic_string();
            } else {
                filename = filepath.stem().generic_string();
            }
        }
        return filename;
    } catch (std::filesystem::filesystem_error const& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
        return std::string();
    }
}


template<typename T, typename S>
std::string megamol::core::utility::FileUtils::SearchFileRecursive(const T& search_path_str, const S& search_file_str) {
    try {
        auto search_path = std::filesystem::u8path(search_path_str);
        auto file_path = std::filesystem::u8path(search_file_str);
        std::string found_path;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(search_path)) {
            if (entry.path().filename() == file_path) {
                found_path = entry.path().generic_string();
                break;
            }
        }
        return found_path;
    } catch (std::filesystem::filesystem_error const& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Filesystem Error: %s [%s, %s, line %d]\n", e.what(), __FILE__, __FUNCTION__, __LINE__);
    }
    return std::string();
}


} // namespace megamol::core::utility
