/*
 * GUIUtils.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GUIUTILS_INCLUDED
#define MEGAMOL_GUI_GUIUTILS_INCLUDED

#include "mmcore/view/Input.h"

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm> // search
#include <cctype>    // toupper
#include <string>
#include <tuple>
#include <utility>

/// CMake exeption for the cluster "stampede2" running CentOS. (C++ filesystem support is not working?)
#ifdef GUI_USE_FILESYSTEM
#    include "FileUtils.h"
#endif // GUI_USE_FILESYSTEM


namespace megamol {
namespace gui {


/********** Defines **********/

#define GUI_INVALID_ID        (UINT_MAX)
#define GUI_CALL_SLOT_RADIUS  (8.0f)
#define GUI_MAX_MULITLINE     (7)


/********** Types **********/

/** Hotkey Data Types for Configurator */
typedef std::tuple<megamol::core::view::KeyCode, bool> HotkeyDataType;

enum HotkeyIndex : size_t { MODULE_SEARCH = 0, PARAMETER_SEARCH = 1, DELETE_GRAPH_ITEM = 2, INDEX_COUNT = 3 };

typedef std::array<HotkeyDataType, HotkeyIndex::INDEX_COUNT> HotKeyArrayType;

/* Canvas Data Type for Information of Graph */
typedef struct _canvas_ {
    ImVec2 position;
    ImVec2 size;
    ImVec2 scrolling;
    float zooming;
    ImVec2 offset;
    bool updated;
} Canvas;


/**
 * Utility class for GUIUtils-style widgets.
 */
class GUIUtils {
public:
    GUIUtils(void);

    ~GUIUtils(void) = default;

    // Tool tip widgets -------------------------------------------------------

    /**
     * Show tooltip on hover.
     *
     * @param text        The tooltip text.
     * @param id          The id of the imgui item the tooltip belongs (only needed for delayed appearance of tooltip).
     * @param time_start  The time delay to wait until the tooltip is shown for a hovered imgui item.
     * @param time_end    The time delay to wait until the tooltip is hidden for a hovered imgui item.
     */
    void HoverToolTip(const std::string& text, ImGuiID id = 0, float time_start = 0.0f, float time_end = 4.0f);

    /**
     * Show help marker text with tooltip on hover.
     *
     * @param text   The help tooltip text.
     * @param label  The visible text for which the tooltip is enabled.
     */
    void HelpMarkerToolTip(const std::string& text, std::string label = "(?)");


    // Pu-up widgets -------------------------------------------------------

    bool MinimalPopUp(const std::string& caption, bool open_popup, const std::string& info_text,
        const std::string& confirm_btn_text, bool& confirmed, const std::string& abort_btn_text, bool& aborted);

    bool RenamePopUp(const std::string& caption, bool open_popup, std::string& rename);


#ifdef GUI_USE_FILESYSTEM

    enum FileBrowserFlag { SAVE, LOAD };
    bool FileBrowserPopUp(FileBrowserFlag flag, const std::string& label, bool open_popup, std::string& inout_filename);

#endif // GUI_USE_FILESYSTEM


    // Misc widgets -------------------------------------------------------

    /**
     * Draw draggable splitter between two child windows, relative to parent window size.
     * https://github.com/ocornut/imgui/issues/319
     */
    enum FixedSplitterSide { LEFT, RIGHT };
    bool VerticalSplitter(FixedSplitterSide fixed_side, float& size_left, float& size_right);


    /** "Point in Circle" Button */
    void PointCircleButton(const std::string& label = "");

    // UTF8 String En-/Decoding -----------------------------------------------

    /** Decode string from UTF-8. */
    bool Utf8Decode(std::string& str) const;

    /** Encode string into UTF-8. */
    bool Utf8Encode(std::string& str) const;


    // String search widget ---------------------------------------------------

    /** Show string serach widget. */
    void StringSearch(const std::string& label, const std::string& help);

    /**
     * Returns true if search string is found in source as a case insensitive substring.
     *
     * @param source   The string to search in.
     * @param search   The string to search for in the source.
     */
    inline bool FindCaseInsensitiveSubstring(const std::string& source, const std::string& search) const {

        auto it = std::search(source.begin(), source.end(), search.begin(), search.end(),
            [](char ch1, char ch2) { return std::toupper(ch1) == std::toupper(ch2); });
        return (it != source.end());
    }

    /** Set keyboard focus to search text input. */
    inline void SetSearchFocus(bool focus) { this->search_focus = focus; }

    /** Set keyboard focus to search text input. */
    inline std::string GetSearchString(void) const { return this->search_string; }


    // Other utility functions ------------------------------------------------

    /**
     * Returns width of text drawn as widget.
     */
    float TextWidgetWidth(const std::string& text) const;

    /**
     * Set/Unset read only widget style.
     */
    void ReadOnlyWigetStyle(bool set);

private:
    /** Current tooltip hover time. */
    float tooltip_time;

    /** Current hovered tooltip item. */
    ImGuiID tooltip_id;

    /** Set focus to search text input. */
    bool search_focus;

    /** Current search string. */
    std::string search_string;

    /** Current rename string. */
    std::string rename_string;

    /** Splitter width for restoring after collapsing.  */
    float splitter_last_width;

#ifdef GUI_USE_FILESYSTEM

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

    bool splitPath(const fsns::path& in_file_path, std::string& out_path, std::string& out_file);
    void validateDirectory(const std::string& path_str);
    void validateFile(const std::string& file_str, GUIUtils::FileBrowserFlag flag);

#endif // GUI_USE_FILESYSTEM
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUIUTILS_INCLUDED