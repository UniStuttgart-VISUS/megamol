/*
 * gui_utils.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#pragma once

#include <climits>

#include <imgui.h>
#include <imgui_internal.h>

#include "mmcore/param/AbstractParamPresentation.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/Input.h"
#include "vislib/math/Ternary.h"


/// #define GUI_VERBOSE


#define GUI_INVALID_ID (UINT_MAX)
#define GUI_SLOT_RADIUS (8.0f * megamol::gui::gui_scaling.Get())
#define GUI_LINE_THICKNESS (3.0f * megamol::gui::gui_scaling.Get())
#define GUI_RECT_CORNER_RADIUS (0.0f)
#define GUI_MAX_MULITLINE (8)
#define GUI_DND_CALLSLOT_UID_TYPE ("DND_CALL")
#define GUI_GRAPH_BORDER (GUI_SLOT_RADIUS * 4.0f)

#define GUI_JSON_TAG_GUI ("GUIState")
#define GUI_JSON_TAG_WINDOW_CONFIGS ("WindowConfigurations")
#define GUI_JSON_TAG_CONFIGURATOR ("ConfiguratorState")
#define GUI_JSON_TAG_GRAPHS ("GraphStates")
#define GUI_JSON_TAG_PROJECT ("Project")
#define GUI_JSON_TAG_MODULES ("Modules")
#define GUI_JSON_TAG_INTERFACES ("Interfaces")
/// GUI_JSON_TAG_GUISTATE_PARAMETERS ("ParameterStates") is defined in AbstractParamPresentation.h

#define GUI_START_TAG_SET_GUI_STATE ("mmSetGUIState([=[")
#define GUI_END_TAG_SET_GUI_STATE ("]=])")
#define GUI_START_TAG_SET_GUI_VISIBILITY ("mmSetGUIVisible(")
#define GUI_END_TAG_SET_GUI_VISIBILITY (")")
#define GUI_START_TAG_SET_GUI_SCALE ("mmSetGUIScale(")
#define GUI_END_TAG_SET_GUI_SCALE (")")

// Global Colors
#define GUI_COLOR_TEXT_ERROR (ImVec4(0.9f, 0.1f, 0.0f, 1.0f))
#define GUI_COLOR_TEXT_WARN (ImVec4(0.75f, 0.75f, 0.0f, 1.0f))

#define GUI_COLOR_BUTTON_MODIFIED (ImVec4(0.75f, 0.0f, 0.25f, 1.0f))
#define GUI_COLOR_BUTTON_MODIFIED_HIGHLIGHT (ImVec4(0.9f, 0.0f, 0.25f, 1.0f))

#define GUI_COLOR_SLOT_CALLER (ImVec4(0.0f, 0.75f, 1.0f, 1.0f))
#define GUI_COLOR_SLOT_CALLEE (ImVec4(0.75f, 0.0f, 1.0f, 1.0f))
#define GUI_COLOR_SLOT_COMPATIBLE (ImVec4(0.5f, 0.9f, 0.25f, 1.0f))
#define GUI_COLOR_SLOT_REQUIRED (ImVec4(0.9f, 0.75f, 0.1f, 1.0f))

#define GUI_COLOR_GROUP_HEADER (ImVec4(0.0f, 0.5f, 0.25f, 1.0f))
#define GUI_COLOR_GROUP_HEADER_HIGHLIGHT (ImVec4(0.0f, 0.75f, 0.5f, 1.0f))

// Texture File Names
#define GUI_FILENAME_FONT_DEFAULT_ROBOTOSANS ("Roboto-Regular.ttf")
#define GUI_FILENAME_FONT_DEFAULT_SOURCECODEPRO ("SourceCodePro-Regular.ttf")
#define GUI_FILENAME_TEXTURE_TRANSPORT_ICON_PLAY ("transport_ctrl_play.png")
#define GUI_FILENAME_TEXTURE_TRANSPORT_ICON_PAUSE ("transport_ctrl_pause.png")
#define GUI_FILENAME_TEXTURE_TRANSPORT_ICON_FAST_FORWARD ("transport_ctrl_fast-forward.png")
#define GUI_FILENAME_TEXTURE_TRANSPORT_ICON_FAST_REWIND ("transport_ctrl_fast-rewind.png")
#define GUI_FILENAME_TEXTURE_VIEWCUBE_ROTATION_ARROW ("viewcube_rotation_arrow.png")
#define GUI_FILENAME_TEXTURE_VIEWCUBE_UP_ARROW ("viewcube_up_arrow.png")
#define GUI_FILENAME_TEXTURE_PROFILING_BUTTON ("profiling_button.png")


#ifdef MEGAMOL_USE_OPENGL

#define GUI_GL_CHECK_ERROR                                                                        \
    {                                                                                             \
        auto err = glGetError();                                                                  \
        if (err != GL_NO_ERROR)                                                                   \
            megamol::core::utility::log::Log::DefaultLog.WriteError(                              \
                "OpenGL Error: %i. [%s, %s, line %d]\n ", err, __FILE__, __FUNCTION__, __LINE__); \
    }

#endif // MEGAMOL_USE_OPENGL


namespace megamol::gui {


/********** Additional Global ImGui Operators ****************************/

namespace {

bool operator==(const ImVec2& left, const ImVec2& right) {
    return ((left.x == right.x) && (left.y == right.y));
}

bool operator!=(const ImVec2& left, const ImVec2& right) {
    return !(left == right);
}

} // namespace


/********** Global Unique ID *********************************************/

/// ! Do not directly change
extern ImGuiID gui_generated_uid;

inline ImGuiID GenerateUniqueID() {
    return (++gui_generated_uid);
}


/********** Global ImGui Context Pointer Counter *************************/

// Only accessed by possible multiple instances of GUIManager
extern unsigned int gui_context_count;


/********** Global Resource Paths ****************************************/

// Resource paths set by GUIManager
extern std::vector<std::string> gui_resource_paths;


/********** Global ImGui Mouse Wheel Value  *************************/

/// XXX IO This value is not available in child windows via io.MouseWheel
extern float gui_mouse_wheel;


/********** Global GUI Scaling Factor ************************************/

// Forward declaration
class GUIManager;

class GUIScaling {
public:
    friend class GUIManager;

    GUIScaling() = default;
    ~GUIScaling() = default;

    float Get() const {
        return this->scale;
    }

    float TransitionFactor() const {
        return (this->scale / this->last_scale);
    }

    bool PendingChange() const {
        return this->pending_change;
    }

private:
    bool ConsumePendingChange() {
        bool current_pending_change = this->pending_change;
        this->pending_change = false;
        return current_pending_change;
    }

    void Set(float s) {
        this->last_scale = this->scale;
        this->scale = std::max(0.0f, s);
        if (this->scale != this->last_scale) {
            this->pending_change = true;
        }
    }

    bool pending_change = false;
    float scale = 1.0f;
    float last_scale = 1.0f;
};

extern GUIScaling gui_scaling;


/********** Types ********************************************************/

// Forward declaration
class CallSlot;
class InterfaceSlot;
typedef std::shared_ptr<megamol::gui::CallSlot> CallSlotPtr_t;
typedef std::shared_ptr<megamol::gui::InterfaceSlot> InterfaceSlotPtr_t;


// Hotkeys
enum HotkeyIndex : size_t {
    HOTKEY_GUI_EXIT_PROGRAM,
    HOTKEY_GUI_PARAMETER_SEARCH,
    HOTKEY_GUI_SAVE_PROJECT,
    HOTKEY_GUI_LOAD_PROJECT,
    HOTKEY_GUI_MENU,
    HOTKEY_GUI_TOGGLE_GRAPH_ENTRY,
    HOTKEY_GUI_TRIGGER_SCREENSHOT,
    HOTKEY_GUI_SHOW_HIDE_GUI,
    HOTKEY_CONFIGURATOR_MODULE_SEARCH,
    HOTKEY_CONFIGURATOR_PARAMETER_SEARCH,
    HOTKEY_CONFIGURATOR_DELETE_GRAPH_ITEM,
    HOTKEY_CONFIGURATOR_SAVE_PROJECT,
    HOTKEY_CONFIGURATOR_LAYOUT_GRAPH
};
struct HotkeyData_t {
    std::string name;
    megamol::core::view::KeyCode keycode;
    bool is_pressed = false;
};
typedef std::map<HotkeyIndex, megamol::gui::HotkeyData_t> HotkeyMap_t;

typedef megamol::core::param::AbstractParamPresentation::Presentation Present_t;
typedef megamol::core::param::AbstractParamPresentation::ParamType ParamType_t;

typedef std::array<float, 5> FontScalingArray_t;

/* Data type holding a pair of uids. */
typedef std::vector<ImGuiID> UIDVector_t;
typedef std::pair<ImGuiID, ImGuiID> UIDPair_t;
typedef std::vector<UIDPair_t> UIDPairVector_t;

typedef std::pair<std::string, std::string> StrPair_t;
typedef std::vector<StrPair_t> StrPairVector_t;

/* Data type holding current group uid and group name pairs. */
typedef std::pair<ImGuiID, std::string> GraphGroupPair_t;
typedef std::vector<megamol::gui::GraphGroupPair_t> GraphGroupPairVector_t;

enum PresentPhase : size_t { INTERACTION = 0, RENDERING = 1 };

/* Data type holding information of graph canvas. */
typedef struct _canvas_ {
    ImVec2 position;      // in
    ImVec2 size;          // in
    ImVec2 scrolling;     // in
    float zooming;        // in
    ImVec2 offset;        // in
    ImFont* gui_font_ptr; // in
} GraphCanvas_t;

/* Data type holding information on graph item interaction. */
typedef struct _interact_state_ {

    ImGuiID button_active_uid;  // in out
    ImGuiID button_hovered_uid; // in out
    bool process_deletion;      // out

    ImGuiID group_selected_uid; // in out
    ImGuiID group_hovered_uid;  // in out
    bool group_layout;          // out

    UIDVector_t modules_selected_uids;               // in out
    ImGuiID module_hovered_uid;                      // in out
    UIDPairVector_t modules_add_group_uids;          // out
    UIDVector_t modules_remove_group_uids;           // out
    bool modules_layout;                             // out
    StrPairVector_t module_rename;                   // out
    vislib::math::Ternary module_graphentry_changed; // out
    ImVec2 module_param_child_position;              // out
    bool module_show_label;                          // in

    ImGuiID call_selected_uid;       // in out
    ImGuiID call_hovered_uid;        // in out
    bool call_show_label;            // in
    bool call_show_slots_label;      // in
    unsigned int call_coloring_mode; // in
    unsigned int call_coloring_map;  // in

    UIDPair_t slot_drag_drop_uids; // in out

    ImGuiID callslot_selected_uid;       // in out
    ImGuiID callslot_hovered_uid;        // in out
    UIDPair_t callslot_add_group_uid;    // in out
    UIDPair_t callslot_remove_group_uid; // in out
    CallSlotPtr_t callslot_compat_ptr;   // in
    bool callslot_show_label;            // in

    ImGuiID interfaceslot_selected_uid;          // in out
    ImGuiID interfaceslot_hovered_uid;           // in out
    InterfaceSlotPtr_t interfaceslot_compat_ptr; // in

    bool parameters_extended_mode; // in

    bool graph_is_running; // in

    bool profiling_pause_update; // in out
    bool profiling_show;         // in out

} GraphItemsInteract_t;

/* Data type holding shared state of graph items. */
typedef struct _graph_item_state_ {
    megamol::gui::GraphCanvas_t canvas;          // (see above)
    megamol::gui::GraphItemsInteract_t interact; // (see above)
    megamol::gui::HotkeyMap_t hotkeys;           // in out
    megamol::gui::GraphGroupPairVector_t groups; // in
} GraphItemsState_t;

/* Data type holding shared state of graphs. */
typedef struct _graph_state_ {
    FontScalingArray_t graph_zoom_font_scalings; // in
    float graph_width;                           // in
    bool show_parameter_sidebar;                 // in
    bool show_profiling_bar;                     // in
    megamol::gui::HotkeyMap_t hotkeys;           // in out
    ImGuiID graph_selected_uid;                  // out
    bool graph_delete;                           // out
    bool configurator_graph_save;                // out
    bool global_graph_save;                      // out
    ImGuiID new_running_graph_uid;               // out
} GraphState_t;

enum class HeaderType { MODULE_GROUP, MODULE, PARAMETER_GROUP };


/********** gui_utils *****************************************************/

/**
 * Static GUI utility functions.
 */
class gui_utils {
public:
    /** Extract string enclosed in predefined tags. */
    static std::string ExtractTaggedString(
        const std::string& str, const std::string& start_tag, const std::string& end_tag) {

        std::string return_str;
        auto start_idx = str.find(start_tag);
        if (start_idx != std::string::npos) {
            auto end_idx = str.find(end_tag, start_idx);
            if ((end_idx != std::string::npos) && (start_idx < end_idx)) {
                start_idx += std::string(start_tag).length();
                return_str = str.substr(start_idx, (end_idx - start_idx));
            }
        }
        return return_str;
    }

    /**
     * Enable/Disable read only widget style.
     */
    static void PushReadOnly(bool set = true) {

        if (set) {
            ImGui::BeginDisabled(set);
        }
    }
    static void PopReadOnly(bool set = true) {

        if (set) {
            ImGui::EndDisabled();
        }
    }

    /**
     * Returns true if search string is found in source as a case insensitive substring.
     *
     * @param source   The string to search in.
     * @param search   The string to search for in the source.
     */
    static bool FindCaseInsensitiveSubstring(const std::string& source, const std::string& search) {

        if (search.empty())
            return true;
        auto it = std::search(source.begin(), source.end(), search.begin(), search.end(),
            [](char ch1, char ch2) { return std::toupper(ch1) == std::toupper(ch2); });
        return (it != source.end());
    }

    /**
     * Returns true if both strings have same length(!) and equal each other case insensitively.
     *
     * @param source   One string.
     * @param search   Second string.
     */
    static bool CaseInsensitiveStringEqual(std::string const& str1, std::string const& str2) {

        return ((str1.size() == str2.size()) &&
                std::equal(str1.begin(), str1.end(), str2.begin(),
                    [](char const& c1, char const& c2) { return (c1 == c2 || std::toupper(c1) == std::toupper(c2)); }));
    }

    /**
     * Returns true if check string is contained in reference string case insensitively.
     *
     * @param ref_str   Reference string.
     * @param chk_str   String to check if contained in reference string.
     */
    static bool CaseInsensitiveStringContain(std::string const& ref_str, std::string const& chk_str) {

        if (ref_str.size() < chk_str.size())
            return false;
        return std::equal(chk_str.begin(), chk_str.end(), ref_str.begin(),
            [](char const& c1, char const& c2) { return (c1 == c2 || std::toupper(c1) == std::toupper(c2)); });
    }

    /*
     * Draw collapsing group header.
     */
    static bool GroupHeader(megamol::gui::HeaderType type, const std::string& name, std::string& inout_search,
        ImGuiID override_header_state = GUI_INVALID_ID) {

        if (ImGui::GetCurrentContext() == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] No ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        // Determine header state and change color depending on active parameter search
        auto headerId = ImGui::GetID(name.c_str());
        auto headerState = override_header_state;
        if (headerState == GUI_INVALID_ID) {
            headerState = ImGui::GetStateStorage()->GetInt(headerId, 0); // 0=close 1=open
        }

        int pop_style_color_number = 0;
        if (type == megamol::gui::HeaderType::MODULE_GROUP) {
            ImGui::PushStyleColor(ImGuiCol_Header, GUI_COLOR_GROUP_HEADER);
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, GUI_COLOR_GROUP_HEADER_HIGHLIGHT);
            auto header_active_color = GUI_COLOR_GROUP_HEADER_HIGHLIGHT;
            header_active_color.w *= 0.75f;
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, header_active_color);
            pop_style_color_number += 3;
        }

        if (!inout_search.empty()) {
            headerState = 1;
            bool searched = gui_utils::FindCaseInsensitiveSubstring(name, inout_search);
            if (!searched) {
                auto header_col = ImGui::GetStyleColorVec4(ImGuiCol_Header);
                header_col.w *= 0.25;
                ImGui::PushStyleColor(ImGuiCol_Header, header_col);
                auto header_hovered_col = ImGui::GetStyleColorVec4(ImGuiCol_HeaderHovered);
                header_hovered_col.w *= 0.25;
                ImGui::PushStyleColor(ImGuiCol_HeaderHovered, header_hovered_col);
                auto text_col = ImGui::GetStyleColorVec4(ImGuiCol_Text);
                text_col.w *= 0.25;
                ImGui::PushStyleColor(ImGuiCol_Text, text_col);
                pop_style_color_number += 3;
            } else {
                // Show all below when given name is part of the search
                inout_search.clear();
            }
        }
        ImGui::GetStateStorage()->SetInt(headerId, static_cast<int>(headerState));
        bool header_open = ImGui::CollapsingHeader(name.c_str(), nullptr);
        ImGui::PopStyleColor(pop_style_color_number);

        // Keep following elements open for one more frame to propagate override changes to headers below.
        if (override_header_state == 0) {
            header_open = true;
        }
        return header_open;
    }

    static ImGuiKey GlfwKeyToImGuiKey(megamol::frontend_resources::Key key) {
        switch (key) {
        case megamol::frontend_resources::Key::KEY_TAB:
            return ImGuiKey_Tab;
        case megamol::frontend_resources::Key::KEY_LEFT:
            return ImGuiKey_LeftArrow;
        case megamol::frontend_resources::Key::KEY_RIGHT:
            return ImGuiKey_RightArrow;
        case megamol::frontend_resources::Key::KEY_UP:
            return ImGuiKey_UpArrow;
        case megamol::frontend_resources::Key::KEY_DOWN:
            return ImGuiKey_DownArrow;
        case megamol::frontend_resources::Key::KEY_PAGE_UP:
            return ImGuiKey_PageUp;
        case megamol::frontend_resources::Key::KEY_PAGE_DOWN:
            return ImGuiKey_PageDown;
        case megamol::frontend_resources::Key::KEY_HOME:
            return ImGuiKey_Home;
        case megamol::frontend_resources::Key::KEY_END:
            return ImGuiKey_End;
        case megamol::frontend_resources::Key::KEY_INSERT:
            return ImGuiKey_Insert;
        case megamol::frontend_resources::Key::KEY_DELETE:
            return ImGuiKey_Delete;
        case megamol::frontend_resources::Key::KEY_BACKSPACE:
            return ImGuiKey_Backspace;
        case megamol::frontend_resources::Key::KEY_SPACE:
            return ImGuiKey_Space;
        case megamol::frontend_resources::Key::KEY_ENTER:
            return ImGuiKey_Enter;
        case megamol::frontend_resources::Key::KEY_ESCAPE:
            return ImGuiKey_Escape;
        case megamol::frontend_resources::Key::KEY_APOSTROPHE:
            return ImGuiKey_Apostrophe;
        case megamol::frontend_resources::Key::KEY_COMMA:
            return ImGuiKey_Comma;
        case megamol::frontend_resources::Key::KEY_MINUS:
            return ImGuiKey_Minus;
        case megamol::frontend_resources::Key::KEY_PERIOD:
            return ImGuiKey_Period;
        case megamol::frontend_resources::Key::KEY_SLASH:
            return ImGuiKey_Slash;
        case megamol::frontend_resources::Key::KEY_SEMICOLON:
            return ImGuiKey_Semicolon;
        case megamol::frontend_resources::Key::KEY_EQUAL:
            return ImGuiKey_Equal;
        case megamol::frontend_resources::Key::KEY_LEFT_BRACKET:
            return ImGuiKey_LeftBracket;
        case megamol::frontend_resources::Key::KEY_BACKSLASH:
            return ImGuiKey_Backslash;
        case megamol::frontend_resources::Key::KEY_RIGHT_BRACKET:
            return ImGuiKey_RightBracket;
        case megamol::frontend_resources::Key::KEY_GRAVE_ACCENT:
            return ImGuiKey_GraveAccent;
        case megamol::frontend_resources::Key::KEY_CAPS_LOCK:
            return ImGuiKey_CapsLock;
        case megamol::frontend_resources::Key::KEY_SCROLL_LOCK:
            return ImGuiKey_ScrollLock;
        case megamol::frontend_resources::Key::KEY_NUM_LOCK:
            return ImGuiKey_NumLock;
        case megamol::frontend_resources::Key::KEY_PRINT_SCREEN:
            return ImGuiKey_PrintScreen;
        case megamol::frontend_resources::Key::KEY_PAUSE:
            return ImGuiKey_Pause;
        case megamol::frontend_resources::Key::KEY_KP_0:
            return ImGuiKey_Keypad0;
        case megamol::frontend_resources::Key::KEY_KP_1:
            return ImGuiKey_Keypad1;
        case megamol::frontend_resources::Key::KEY_KP_2:
            return ImGuiKey_Keypad2;
        case megamol::frontend_resources::Key::KEY_KP_3:
            return ImGuiKey_Keypad3;
        case megamol::frontend_resources::Key::KEY_KP_4:
            return ImGuiKey_Keypad4;
        case megamol::frontend_resources::Key::KEY_KP_5:
            return ImGuiKey_Keypad5;
        case megamol::frontend_resources::Key::KEY_KP_6:
            return ImGuiKey_Keypad6;
        case megamol::frontend_resources::Key::KEY_KP_7:
            return ImGuiKey_Keypad7;
        case megamol::frontend_resources::Key::KEY_KP_8:
            return ImGuiKey_Keypad8;
        case megamol::frontend_resources::Key::KEY_KP_9:
            return ImGuiKey_Keypad9;
        case megamol::frontend_resources::Key::KEY_KP_DECIMAL:
            return ImGuiKey_KeypadDecimal;
        case megamol::frontend_resources::Key::KEY_KP_DIVIDE:
            return ImGuiKey_KeypadDivide;
        case megamol::frontend_resources::Key::KEY_KP_MULTIPLY:
            return ImGuiKey_KeypadMultiply;
        case megamol::frontend_resources::Key::KEY_KP_SUBTRACT:
            return ImGuiKey_KeypadSubtract;
        case megamol::frontend_resources::Key::KEY_KP_ADD:
            return ImGuiKey_KeypadAdd;
        case megamol::frontend_resources::Key::KEY_KP_ENTER:
            return ImGuiKey_KeypadEnter;
        case megamol::frontend_resources::Key::KEY_KP_EQUAL:
            return ImGuiKey_KeypadEqual;
        case megamol::frontend_resources::Key::KEY_LEFT_SHIFT:
            return ImGuiKey_LeftShift;
        case megamol::frontend_resources::Key::KEY_LEFT_CONTROL:
            return ImGuiKey_LeftCtrl;
        case megamol::frontend_resources::Key::KEY_LEFT_ALT:
            return ImGuiKey_LeftAlt;
        case megamol::frontend_resources::Key::KEY_LEFT_SUPER:
            return ImGuiKey_LeftSuper;
        case megamol::frontend_resources::Key::KEY_RIGHT_SHIFT:
            return ImGuiKey_RightShift;
        case megamol::frontend_resources::Key::KEY_RIGHT_CONTROL:
            return ImGuiKey_RightCtrl;
        case megamol::frontend_resources::Key::KEY_RIGHT_ALT:
            return ImGuiKey_RightAlt;
        case megamol::frontend_resources::Key::KEY_RIGHT_SUPER:
            return ImGuiKey_RightSuper;
        case megamol::frontend_resources::Key::KEY_MENU:
            return ImGuiKey_Menu;
        case megamol::frontend_resources::Key::KEY_0:
            return ImGuiKey_0;
        case megamol::frontend_resources::Key::KEY_1:
            return ImGuiKey_1;
        case megamol::frontend_resources::Key::KEY_2:
            return ImGuiKey_2;
        case megamol::frontend_resources::Key::KEY_3:
            return ImGuiKey_3;
        case megamol::frontend_resources::Key::KEY_4:
            return ImGuiKey_4;
        case megamol::frontend_resources::Key::KEY_5:
            return ImGuiKey_5;
        case megamol::frontend_resources::Key::KEY_6:
            return ImGuiKey_6;
        case megamol::frontend_resources::Key::KEY_7:
            return ImGuiKey_7;
        case megamol::frontend_resources::Key::KEY_8:
            return ImGuiKey_8;
        case megamol::frontend_resources::Key::KEY_9:
            return ImGuiKey_9;
        case megamol::frontend_resources::Key::KEY_A:
            return ImGuiKey_A;
        case megamol::frontend_resources::Key::KEY_B:
            return ImGuiKey_B;
        case megamol::frontend_resources::Key::KEY_C:
            return ImGuiKey_C;
        case megamol::frontend_resources::Key::KEY_D:
            return ImGuiKey_D;
        case megamol::frontend_resources::Key::KEY_E:
            return ImGuiKey_E;
        case megamol::frontend_resources::Key::KEY_F:
            return ImGuiKey_F;
        case megamol::frontend_resources::Key::KEY_G:
            return ImGuiKey_G;
        case megamol::frontend_resources::Key::KEY_H:
            return ImGuiKey_H;
        case megamol::frontend_resources::Key::KEY_I:
            return ImGuiKey_I;
        case megamol::frontend_resources::Key::KEY_J:
            return ImGuiKey_J;
        case megamol::frontend_resources::Key::KEY_K:
            return ImGuiKey_K;
        case megamol::frontend_resources::Key::KEY_L:
            return ImGuiKey_L;
        case megamol::frontend_resources::Key::KEY_M:
            return ImGuiKey_M;
        case megamol::frontend_resources::Key::KEY_N:
            return ImGuiKey_N;
        case megamol::frontend_resources::Key::KEY_O:
            return ImGuiKey_O;
        case megamol::frontend_resources::Key::KEY_P:
            return ImGuiKey_P;
        case megamol::frontend_resources::Key::KEY_Q:
            return ImGuiKey_Q;
        case megamol::frontend_resources::Key::KEY_R:
            return ImGuiKey_R;
        case megamol::frontend_resources::Key::KEY_S:
            return ImGuiKey_S;
        case megamol::frontend_resources::Key::KEY_T:
            return ImGuiKey_T;
        case megamol::frontend_resources::Key::KEY_U:
            return ImGuiKey_U;
        case megamol::frontend_resources::Key::KEY_V:
            return ImGuiKey_V;
        case megamol::frontend_resources::Key::KEY_W:
            return ImGuiKey_W;
        case megamol::frontend_resources::Key::KEY_X:
            return ImGuiKey_X;
        case megamol::frontend_resources::Key::KEY_Y:
            return ImGuiKey_Y;
        case megamol::frontend_resources::Key::KEY_Z:
            return ImGuiKey_Z;
        case megamol::frontend_resources::Key::KEY_F1:
            return ImGuiKey_F1;
        case megamol::frontend_resources::Key::KEY_F2:
            return ImGuiKey_F2;
        case megamol::frontend_resources::Key::KEY_F3:
            return ImGuiKey_F3;
        case megamol::frontend_resources::Key::KEY_F4:
            return ImGuiKey_F4;
        case megamol::frontend_resources::Key::KEY_F5:
            return ImGuiKey_F5;
        case megamol::frontend_resources::Key::KEY_F6:
            return ImGuiKey_F6;
        case megamol::frontend_resources::Key::KEY_F7:
            return ImGuiKey_F7;
        case megamol::frontend_resources::Key::KEY_F8:
            return ImGuiKey_F8;
        case megamol::frontend_resources::Key::KEY_F9:
            return ImGuiKey_F9;
        case megamol::frontend_resources::Key::KEY_F10:
            return ImGuiKey_F10;
        case megamol::frontend_resources::Key::KEY_F11:
            return ImGuiKey_F11;
        case megamol::frontend_resources::Key::KEY_F12:
            return ImGuiKey_F12;
        default:
            return ImGuiKey_None;
        }
    }


    static megamol::frontend_resources::Key ImGuiKeyToGlfwKey(ImGuiKey key) {
        switch (key) {
        case ImGuiKey_Tab:
            return megamol::frontend_resources::Key::KEY_TAB;
        case ImGuiKey_LeftArrow:
            return megamol::frontend_resources::Key::KEY_LEFT;
        case ImGuiKey_RightArrow:
            return megamol::frontend_resources::Key::KEY_RIGHT;
        case ImGuiKey_UpArrow:
            return megamol::frontend_resources::Key::KEY_UP;
        case ImGuiKey_DownArrow:
            return megamol::frontend_resources::Key::KEY_DOWN;
        case ImGuiKey_PageUp:
            return megamol::frontend_resources::Key::KEY_PAGE_UP;
        case ImGuiKey_PageDown:
            return megamol::frontend_resources::Key::KEY_PAGE_DOWN;
        case ImGuiKey_Home:
            return megamol::frontend_resources::Key::KEY_HOME;
        case ImGuiKey_End:
            return megamol::frontend_resources::Key::KEY_END;
        case ImGuiKey_Insert:
            return megamol::frontend_resources::Key::KEY_INSERT;
        case ImGuiKey_Delete:
            return megamol::frontend_resources::Key::KEY_DELETE;
        case ImGuiKey_Backspace:
            return megamol::frontend_resources::Key::KEY_BACKSPACE;
        case ImGuiKey_Space:
            return megamol::frontend_resources::Key::KEY_SPACE;
        case ImGuiKey_Enter:
            return megamol::frontend_resources::Key::KEY_ENTER;
        case ImGuiKey_Escape:
            return megamol::frontend_resources::Key::KEY_ESCAPE;
        case ImGuiKey_Apostrophe:
            return megamol::frontend_resources::Key::KEY_APOSTROPHE;
        case ImGuiKey_Comma:
            return megamol::frontend_resources::Key::KEY_COMMA;
        case ImGuiKey_Minus:
            return megamol::frontend_resources::Key::KEY_MINUS;
        case ImGuiKey_Period:
            return megamol::frontend_resources::Key::KEY_PERIOD;
        case ImGuiKey_Slash:
            return megamol::frontend_resources::Key::KEY_SLASH;
        case ImGuiKey_Semicolon:
            return megamol::frontend_resources::Key::KEY_SEMICOLON;
        case ImGuiKey_Equal:
            return megamol::frontend_resources::Key::KEY_EQUAL;
        case ImGuiKey_LeftBracket:
            return megamol::frontend_resources::Key::KEY_LEFT_BRACKET;
        case ImGuiKey_Backslash:
            return megamol::frontend_resources::Key::KEY_BACKSLASH;
        case ImGuiKey_RightBracket:
            return megamol::frontend_resources::Key::KEY_RIGHT_BRACKET;
        case ImGuiKey_GraveAccent:
            return megamol::frontend_resources::Key::KEY_GRAVE_ACCENT;
        case ImGuiKey_CapsLock:
            return megamol::frontend_resources::Key::KEY_CAPS_LOCK;
        case ImGuiKey_ScrollLock:
            return megamol::frontend_resources::Key::KEY_SCROLL_LOCK;
        case ImGuiKey_NumLock:
            return megamol::frontend_resources::Key::KEY_NUM_LOCK;
        case ImGuiKey_PrintScreen:
            return megamol::frontend_resources::Key::KEY_PRINT_SCREEN;
        case ImGuiKey_Pause:
            return megamol::frontend_resources::Key::KEY_PAUSE;
        case ImGuiKey_Keypad0:
            return megamol::frontend_resources::Key::KEY_KP_0;
        case ImGuiKey_Keypad1:
            return megamol::frontend_resources::Key::KEY_KP_1;
        case ImGuiKey_Keypad2:
            return megamol::frontend_resources::Key::KEY_KP_2;
        case ImGuiKey_Keypad3:
            return megamol::frontend_resources::Key::KEY_KP_3;
        case ImGuiKey_Keypad4:
            return megamol::frontend_resources::Key::KEY_KP_4;
        case ImGuiKey_Keypad5:
            return megamol::frontend_resources::Key::KEY_KP_5;
        case ImGuiKey_Keypad6:
            return megamol::frontend_resources::Key::KEY_KP_6;
        case ImGuiKey_Keypad7:
            return megamol::frontend_resources::Key::KEY_KP_7;
        case ImGuiKey_Keypad8:
            return megamol::frontend_resources::Key::KEY_KP_8;
        case ImGuiKey_Keypad9:
            return megamol::frontend_resources::Key::KEY_KP_9;
        case ImGuiKey_KeypadDecimal:
            return megamol::frontend_resources::Key::KEY_KP_DECIMAL;
        case ImGuiKey_KeypadDivide:
            return megamol::frontend_resources::Key::KEY_KP_DIVIDE;
        case ImGuiKey_KeypadMultiply:
            return megamol::frontend_resources::Key::KEY_KP_MULTIPLY;
        case ImGuiKey_KeypadSubtract:
            return megamol::frontend_resources::Key::KEY_KP_SUBTRACT;
        case ImGuiKey_KeypadAdd:
            return megamol::frontend_resources::Key::KEY_KP_ADD;
        case ImGuiKey_KeypadEnter:
            return megamol::frontend_resources::Key::KEY_KP_ENTER;
        case ImGuiKey_KeypadEqual:
            return megamol::frontend_resources::Key::KEY_KP_EQUAL;
        case ImGuiKey_LeftShift:
            return megamol::frontend_resources::Key::KEY_LEFT_SHIFT;
        case ImGuiKey_LeftCtrl:
            return megamol::frontend_resources::Key::KEY_LEFT_CONTROL;
        case ImGuiKey_LeftAlt:
            return megamol::frontend_resources::Key::KEY_LEFT_ALT;
        case ImGuiKey_LeftSuper:
            return megamol::frontend_resources::Key::KEY_LEFT_SUPER;
        case ImGuiKey_RightShift:
            return megamol::frontend_resources::Key::KEY_RIGHT_SHIFT;
        case ImGuiKey_RightCtrl:
            return megamol::frontend_resources::Key::KEY_RIGHT_CONTROL;
        case ImGuiKey_RightAlt:
            return megamol::frontend_resources::Key::KEY_RIGHT_ALT;
        case ImGuiKey_RightSuper:
            return megamol::frontend_resources::Key::KEY_RIGHT_SUPER;
        case ImGuiKey_Menu:
            return megamol::frontend_resources::Key::KEY_MENU;
        case ImGuiKey_0:
            return megamol::frontend_resources::Key::KEY_0;
        case ImGuiKey_1:
            return megamol::frontend_resources::Key::KEY_1;
        case ImGuiKey_2:
            return megamol::frontend_resources::Key::KEY_2;
        case ImGuiKey_3:
            return megamol::frontend_resources::Key::KEY_3;
        case ImGuiKey_4:
            return megamol::frontend_resources::Key::KEY_4;
        case ImGuiKey_5:
            return megamol::frontend_resources::Key::KEY_5;
        case ImGuiKey_6:
            return megamol::frontend_resources::Key::KEY_6;
        case ImGuiKey_7:
            return megamol::frontend_resources::Key::KEY_7;
        case ImGuiKey_8:
            return megamol::frontend_resources::Key::KEY_8;
        case ImGuiKey_9:
            return megamol::frontend_resources::Key::KEY_9;
        case ImGuiKey_A:
            return megamol::frontend_resources::Key::KEY_A;
        case ImGuiKey_B:
            return megamol::frontend_resources::Key::KEY_B;
        case ImGuiKey_C:
            return megamol::frontend_resources::Key::KEY_C;
        case ImGuiKey_D:
            return megamol::frontend_resources::Key::KEY_D;
        case ImGuiKey_E:
            return megamol::frontend_resources::Key::KEY_E;
        case ImGuiKey_F:
            return megamol::frontend_resources::Key::KEY_F;
        case ImGuiKey_G:
            return megamol::frontend_resources::Key::KEY_G;
        case ImGuiKey_H:
            return megamol::frontend_resources::Key::KEY_H;
        case ImGuiKey_I:
            return megamol::frontend_resources::Key::KEY_I;
        case ImGuiKey_J:
            return megamol::frontend_resources::Key::KEY_J;
        case ImGuiKey_K:
            return megamol::frontend_resources::Key::KEY_K;
        case ImGuiKey_L:
            return megamol::frontend_resources::Key::KEY_L;
        case ImGuiKey_M:
            return megamol::frontend_resources::Key::KEY_M;
        case ImGuiKey_N:
            return megamol::frontend_resources::Key::KEY_N;
        case ImGuiKey_O:
            return megamol::frontend_resources::Key::KEY_O;
        case ImGuiKey_P:
            return megamol::frontend_resources::Key::KEY_P;
        case ImGuiKey_Q:
            return megamol::frontend_resources::Key::KEY_Q;
        case ImGuiKey_R:
            return megamol::frontend_resources::Key::KEY_R;
        case ImGuiKey_S:
            return megamol::frontend_resources::Key::KEY_S;
        case ImGuiKey_T:
            return megamol::frontend_resources::Key::KEY_T;
        case ImGuiKey_U:
            return megamol::frontend_resources::Key::KEY_U;
        case ImGuiKey_V:
            return megamol::frontend_resources::Key::KEY_V;
        case ImGuiKey_W:
            return megamol::frontend_resources::Key::KEY_W;
        case ImGuiKey_X:
            return megamol::frontend_resources::Key::KEY_X;
        case ImGuiKey_Y:
            return megamol::frontend_resources::Key::KEY_Y;
        case ImGuiKey_Z:
            return megamol::frontend_resources::Key::KEY_Z;
        case ImGuiKey_F1:
            return megamol::frontend_resources::Key::KEY_F1;
        case ImGuiKey_F2:
            return megamol::frontend_resources::Key::KEY_F2;
        case ImGuiKey_F3:
            return megamol::frontend_resources::Key::KEY_F3;
        case ImGuiKey_F4:
            return megamol::frontend_resources::Key::KEY_F4;
        case ImGuiKey_F5:
            return megamol::frontend_resources::Key::KEY_F5;
        case ImGuiKey_F6:
            return megamol::frontend_resources::Key::KEY_F6;
        case ImGuiKey_F7:
            return megamol::frontend_resources::Key::KEY_F7;
        case ImGuiKey_F8:
            return megamol::frontend_resources::Key::KEY_F8;
        case ImGuiKey_F9:
            return megamol::frontend_resources::Key::KEY_F9;
        case ImGuiKey_F10:
            return megamol::frontend_resources::Key::KEY_F10;
        case ImGuiKey_F11:
            return megamol::frontend_resources::Key::KEY_F11;
        case ImGuiKey_F12:
            return megamol::frontend_resources::Key::KEY_F12;
        default:
            return megamol::frontend_resources::Key::KEY_UNKNOWN;
        }
    }

private:
    gui_utils() = default;
    ~gui_utils() = default;
};


} // namespace megamol::gui
