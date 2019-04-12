/*
 * GUIUtility.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#ifndef MEGAMOL_GUI_GUIUTILITY_H_INCLUDED
#define MEGAMOL_GUI_GUIUTILITY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <cassert>
#include <string>

#include <imgui.h>

#if _HAS_CXX17
#    include <filesystem> // directory_iterator
namespace ns_fs = std::filesystem;
#else
// WINDOWS
#    ifdef _WIN32
#        include <filesystem>
#    else
// LINUX
#        include <experimental/filesystem>
#    endif
namespace ns_fs = std::experimental::filesystem;
#endif


namespace megamol {
namespace gui {


/**
 * Utility functions for the GUI
 */
class GUIUtility {

public:
    typedef ns_fs::path PathType;

    /**
     * Check if given file or directory exists.
     *
     * @param path  The file or directory path.
     */
    bool FilePathExists(std::string path);
    bool FilePathExists(std::wstring path);
    bool FilePathExists(PathType path);

    /**
     * Check if filename exists and has specified file extension.
     *
     * @param path  The file or directory path.
     * @param ext   The extension the given file should have.
     */
    bool FileHasExtension(std::string path, std::string ext);
    bool FileHasExtension(std::wstring path, std::string ext);
    bool FileHasExtension(PathType path, std::string ext);

    /**
     * Search recursively for file or path beginning at given directory.
     *
     * @param path          The file or directory to search for.
     * @param search_path   The path of a directory as start for recursive search.
     */
    bool SearchFilePathRecursive(std::string path, std::string search_path);
    bool SearchFilePathRecursive(std::string path, std::wstring search_path);
    bool SearchFilePathRecursive(std::string path, PathType search_path);

    /**
     * Show tooltip on hover.
     *
     * @param text        The tooltip text.
     * @param id          The id of the imgui item the tooltip belongs (only needed for delayed appearance of tooltip).
     * @param time_start  The time delay to wait until the tooltip is shown for a hovered imgui item.
     * @param time_end    The time delay to wait until the tooltip is hidden for a hovered imgui item.
     */
    void HoverToolTip(std::string text, ImGuiID id = 0, float time_start = 0.0f, float time_end = 4.0f);

    /**
     * Show help marker text with tooltip on hover.
     *
     * @param text   The help tooltip text.
     * @param label  The visible text for which the tooltip is enabled.
     */
    void HelpMarkerToolTip(std::string text, std::string label = "(?)");

protected:
    /**
     * Ctor
     */
    GUIUtility(void);

    /**
     * Dtor
     */
    ~GUIUtility(void);

private:
    /** Current tooltip hover time. */
    float tooltip_time;

    /** Current hovered tooltip item. */
    ImGuiID tooltip_id;
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUIUTILITY_H_INCLUDED
