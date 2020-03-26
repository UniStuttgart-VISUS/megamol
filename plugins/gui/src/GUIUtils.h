/*
 * GUIUtils.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GUIUTILS_INCLUDED
#define MEGAMOL_GUI_GUIUTILS_INCLUDED


#include "mmcore/view/Input.h"

#include "vislib/sys/Log.h"

#define IMGUI_DISABLE_OBSOLETE_FUNCTIONS
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
#include <map>
#include <memory>
#include <vector>

namespace megamol {
namespace gui {

namespace configurator {

// Forward declaration
class Call;
class CallSlot;
class Module;
class Parameter;

// Pointer types to classes
typedef std::shared_ptr<Parameter> ParamPtrType;
typedef std::shared_ptr<Call> CallPtrType;
typedef std::shared_ptr<CallSlot> CallSlotPtrType;
typedef std::shared_ptr<Module> ModulePtrType;

}

/********** Defines **********/

#define GUI_INVALID_ID (UINT_MAX)
#define GUI_CALL_SLOT_RADIUS (8.0f)
#define GUI_MAX_MULITLINE (7)
#define GUI_DND_CALL_UID_TYPE ("DND_CALL")

/********** Types **********/

/** Hotkey Data Types for Configurator */
typedef std::tuple<megamol::core::view::KeyCode, bool> HotkeyDataType;

enum HotkeyIndex : size_t { MODULE_SEARCH = 0, PARAMETER_SEARCH = 1, DELETE_GRAPH_ITEM = 2, SAVE_PROJECT = 3, INDEX_COUNT = 4 };

typedef std::array<HotkeyDataType, HotkeyIndex::INDEX_COUNT> HotKeyArrayType;

/* Data type holding information of graph canvas. */
typedef struct _canvas_ {
    ImVec2 position;
    ImVec2 size;
    ImVec2 scrolling;
    float zooming;
    ImVec2 offset;
} CanvasType;

/* Data type holding information on call slot interaction */
typedef struct _interact_state_ {
    ImGuiID item_selected_uid;
    ImGuiID module_hovered_uid;
    ImGuiID callslot_hovered_uid;
    ImGuiID callslot_dropped_uid;
    configurator::CallSlotPtrType in_compat_slot_ptr;
} InteractType;

typedef struct _state_ {
    CanvasType canvas;
    InteractType interact;
    HotKeyArrayType hotkeys;
} StateType;


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


    // Misc widgets -------------------------------------------------------

    /**
     * Draw draggable splitter between two child windows, relative to parent window size.
     * https://github.com/ocornut/imgui/issues/319
     */
    enum FixedSplitterSide { LEFT, RIGHT };
    bool VerticalSplitter(FixedSplitterSide fixed_side, float& size_left, float& size_right);

    /** "Point in Circle" Button */
    bool PointCircleButton(const std::string& label = "");


    // Static UTF8 String En-/Decoding ----------------------------------------

    /** Decode string from UTF-8. */
    static bool Utf8Decode(std::string& str);

    /** Encode string into UTF-8. */
    static bool Utf8Encode(std::string& str);


    // String search widget ---------------------------------------------------

    /** Show string serach widget. */
    bool StringSearch(const std::string& label, const std::string& help);

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
    // VARIABLES --------------------------------------------------------------

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
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUIUTILS_INCLUDED