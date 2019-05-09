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


#define GUI_MAX_BUFFER_LEN (2048)


namespace megamol {
namespace gui {


/**
 * Utility functions for the GUI
 */
class GUIUtility {

public:
    // FILESYSTEM -------------------------------------------------------------

    /** Type for filesystem paths. */
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
     * @param file          The file to search for.
     * @param search_path   The path of a directory as start for recursive search.
     *
     * @return              The complete path of the found file, empty string otherwise.
     */
    std::string SearchFilePathRecursive(std::string file, std::string search_path);
    std::string SearchFilePathRecursive(std::string file, std::wstring search_path);
    std::string SearchFilePathRecursive(std::string file, PathType search_path);

    // TOOLTIP ----------------------------------------------------------------

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

    // POPUP ------------------------------------------------------------------

    /**
     * Open PopUp asking for user input.
     *
     * @param popup_name   The popup title.
     * @param request      The descriptopn of the requested text input (e.g. file name).
     * @param open         The flag indicating that the popup should be opened.
     *
     * @preturn The captured text input.
     */
    std::string InputDialogPopUp(std::string popup_name, std::string request, bool open);

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
