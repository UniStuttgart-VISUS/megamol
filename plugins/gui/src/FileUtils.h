/*
 * FileUtils.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
/// There is a CMake exeption for the cluster "stampede2" running CentOS, which undefines GUI_USE_FILESYSTEM.


#ifndef MEGAMOL_GUI_FILEUTILS_INCLUDED
#define MEGAMOL_GUI_FILEUTILS_INCLUDED

#ifdef GUI_USE_FILESYSTEM
#    if defined(_HAS_CXX17) || ((defined(_MSC_VER) && (_MSC_VER > 1916))) // C++2017 or since VS2019
#        include <filesystem>
namespace fsns = std::filesystem;
#    else
// WINDOWS
#        ifdef _WIN32
#            include <filesystem>
namespace fsns = std::experimental::filesystem;
#        else
// LINUX
#            include <experimental/filesystem>
namespace fsns = std::experimental::filesystem;
#        endif
#    endif
#endif // GUI_USE_FILESYSTEM

#include "mmcore/CoreInstance.h"

#include <fstream>
#include <iostream>
#include <string>

#include "GUIUtils.h"


namespace megamol {
namespace gui {

class FileUtils {
public:
    FileUtils(void);

    ~FileUtils(void) = default;

    enum FileBrowserFlag { SAVE, LOAD, SELECT };

    // Static functions -------------------------------------------------------

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
     * Save currently loaded project to lua file.
     *
     * @param project_filename The file name for the project.
     * @param core_instance    The pointer to the core instance.
     *
     * @return True on success, false otherwise.
     */
    static bool SaveProjectFile(const std::string& project_filename, megamol::core::CoreInstance* core_instance);

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

    // Non-static functions ----------------------------------------------------

    /**
     * ImGui file browser pop-up.
     *
     * @param flag                Flag inidicating intention of file browser dialog.
     * @param label               File browser label.
     * @param open_popup          Flag once(!) indicates opening of pop-up.
     * @param inout_filename      The file name of the file.
     *
     * @return True on success, false otherwise.
     */

    bool FileBrowserPopUp(FileBrowserFlag flag, const std::string& label, bool open_popup, std::string& out_filename);

    /**
     * ImGui file browser button opening a file browser pop-up.
     *
     * @param out_filename      The file name of the file.
     *
     * @return True on success, false otherwise.
     */
    bool FileBrowserButton(std::string& out_filename);

private:
#ifdef GUI_USE_FILESYSTEM
    // VARIABLES --------------------------------------------------------------

    GUIUtils utils;
    std::string file_name_str;
    std::string file_path_str;
    bool path_changed;
    bool valid_directory;
    bool valid_file;
    bool valid_ending;
    std::string file_error;
    std::string file_warning;
    // Keeps child path and flag whether child is director or not
    typedef std::pair<fsns::path, bool> ChildDataType;
    std::vector<ChildDataType> child_paths;
    size_t additional_lines;

    // FUNCTIONS --------------------------------------------------------------

    bool splitPath(const fsns::path& in_file_path, std::string& out_path, std::string& out_file);
    void validateDirectory(const std::string& path_str);
    void validateFile(const std::string& file_str, FileBrowserFlag flag);

#endif // GUI_USE_FILESYSTEM
};


template <typename T> bool megamol::gui::FileUtils::FilesExistingExtension(const T& path_str, const std::string& ext) {
#ifdef GUI_USE_FILESYSTEM
    auto path = static_cast<fsns::path>(path_str);
    if (!fsns::exists(path) || !fsns::is_regular_file(path)) {
        return false;
    }
    return (path.extension().generic_u8string() == ext);
#else
    return false;
#endif // GUI_USE_FILESYSTEM
}


template <typename T> bool megamol::gui::FileUtils::FileExtension(const T& path_str, const std::string& ext) {
#ifdef GUI_USE_FILESYSTEM
    auto path = static_cast<fsns::path>(path_str);
    return (path.extension().generic_u8string() == ext);
#else
    return false;
#endif // GUI_USE_FILESYSTEM
}


template <typename T, typename S>
std::string megamol::gui::FileUtils::SearchFileRecursive(const T& search_path_str, const S& search_file_str) {
#ifdef GUI_USE_FILESYSTEM
    auto search_path = static_cast<fsns::path>(search_path_str);
    auto file_path = static_cast<fsns::path>(search_file_str);
    std::string found_path;
    for (const auto& entry : fsns::recursive_directory_iterator(search_path)) {
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
