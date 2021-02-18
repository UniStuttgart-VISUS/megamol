/*
 * FileUtils.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_FILEUTILS_INCLUDED
#define MEGAMOL_GUI_FILEUTILS_INCLUDED

#if defined(_HAS_CXX17) || ((defined(_MSC_VER) && (_MSC_VER > 1916))) // C++2017 or since VS2019
#include <filesystem>
namespace stdfs = std::filesystem;
#else
// WINDOWS
#ifdef _WIN32
#include <filesystem>
namespace stdfs = std::experimental::filesystem;
#else
// LINUX
#include <experimental/filesystem>
namespace stdfs = std::experimental::filesystem;
#endif
#endif

#include <fstream>
#include <istream>
#include <iostream>
#include <codecvt>
#include <locale>
#include <string>

#include "mmcore/utility/log/Log.h"


namespace megamol {
namespace core {
namespace utility {

    // #### Utility string conversion functions ############################ //

    static inline std::string to_string(std::wstring wstr) {
         return std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(wstr);
    }

    static inline std::wstring to_wstring(std::string str) {
        return std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(str);
    }

    // ##################################################################### //
    /**
     * File utility functions.
     */
    class FileUtils {
    public:
        /**
         * Load raw data from file (e.g. texture data)
         */
        static bool LoadRawFile(const std::wstring& filename, std::vector<char>& out_data) {
            return megamol::core::utility::FileUtils::LoadRawFile(
                megamol::core::utility::to_string(filename), out_data);
        }

        static bool LoadRawFile(const std::string& filename, std::vector<char>& out_data);

        /**
         * Check if file exists.
         *
         * @param path  The file or directory path.
         */
        template<typename T>
        static bool FileExists(const T& path_str);

        /**
         * Check if any file exists and has specified file extension.
         *
         * @param path  The file or directory path.
         * @param ext   The extension the given file should have.
         */
        template<typename T>
        static bool FileWithExtensionExists(const T& path_str, const std::string& ext);

        /**
         * Check if any file exists and has specified file extension.
         *
         * @param path  The file or directory path.
         * @param ext   The extension the given file should have.
         */
        template<typename T>
        static bool FileHasExtension(const T& path_str, const std::string& ext);

        /**
         * Get stem of filename (filename without leading path and extension).
         *
         * @param path  The file or directory path.
         */
        template<typename T>
        static std::string GetFilenameStem(const T& path_str);

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
        static bool WriteFile(const std::string& filename, const std::string& in_content, bool silent = false);

        /**
         * Read content from file.
         *
         * @param filename      The file name of the file.
         * @param out_content   The content to read from file.
         * @param silent        Disable log output.
         *
         * @return True on success, false otherwise.
         */
        static bool ReadFile(const std::string& filename, std::string& out_content, bool silent = false);

    private:
        FileUtils(void);
        ~FileUtils(void) = default;
    };


    template<typename T>
    bool megamol::core::utility::FileUtils::FileExists(const T& path_str) {
        auto path = static_cast<stdfs::path>(path_str);
        try {
            if (stdfs::exists(path) && stdfs::is_regular_file(path)) {
                return true;
            }
        } catch (...) {}
        return false;
    }


    template<typename T>
    bool megamol::core::utility::FileUtils::FileWithExtensionExists(const T& path_str, const std::string& ext) {
        if (FileUtils::FileExists<T>(path_str)) {
            auto path = static_cast<stdfs::path>(path_str);
            return (path.extension().generic_u8string() == std::string("." + ext));
        }
        return false;
    }


    template<typename T>
    bool megamol::core::utility::FileUtils::FileHasExtension(const T& path_str, const std::string& ext) {
        auto path = static_cast<stdfs::path>(path_str);
        return (path.extension().generic_u8string() == ext);
    }


    template<typename T>
    std::string megamol::core::utility::FileUtils::GetFilenameStem(const T& path_str) {
        auto path = static_cast<stdfs::path>(path_str);
        std::string filename;
        if (path.has_stem()) {
            filename = path.stem().generic_u8string();
        }
        return filename;
    }


    template<typename T, typename S>
    std::string megamol::core::utility::FileUtils::SearchFileRecursive(const T& search_path_str, const S& search_file_str) {
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
    }


} // namespace utility
} // namespace core
} // namespace megamol

#endif // MEGAMOL_GUI_FILEUTILS_INCLUDED
