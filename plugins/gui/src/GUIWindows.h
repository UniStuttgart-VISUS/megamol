/*
 * GUIWindows.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GUIWINDOWS_H_INCLUDED
#define MEGAMOL_GUI_GUIWINDOWS_H_INCLUDED


#ifdef _WIN32
#ifdef GUI_EXPORTS
#define GUI_API __declspec(dllexport)
#else
#define GUI_API __declspec(dllimport)
#endif
#else // _WIN32
#define GUI_API
#endif // _WIN32


#include "Configurator.h"
#include "CorporateGreyStyle.h"
#include "CorporateWhiteStyle.h"
#include "DefaultStyle.h"
#include "LogConsole.h"
#include "WindowCollection.h"
#include "widgets/FileBrowserWidget.h"
#include "widgets/HoverToolTip.h"
#include "widgets/MinimalPopUp.h"
#include "widgets/StringSearchWidget.h"
#include "widgets/TransferFunctionEditor.h"
#include "widgets/WidgetPicking_gl.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/MegaMolGraph.h"
#include "mmcore/ViewDescription.h"
#include "mmcore/utility/FileUtils.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/versioninfo.h"
#include "mmcore/view/AbstractView_EventConsumption.h"

#include "vislib/math/Rectangle.h"

#include <ctime>
#include <iomanip>
#include <sstream>


namespace megamol {
namespace gui {

    class GUI_API GUIWindows {
    public:
        /**
         * CTOR.
         */
        GUIWindows(void);

        /**
         * DTOR.
         */
        virtual ~GUIWindows(void);

        /**
         * Create ImGui context using OpenGL.
         *
         * @param core_instance     The currently available core instance.
         */
        bool CreateContext(GUIImGuiAPI imgui_api, megamol::core::CoreInstance* core_instance);

        /**
         * Setup and enable ImGui context for subsequent use.
         *
         * @param framebuffer_size   The currently available size of the framebuffer.
         * @param window_size        The currently available size of the window.
         * @param instance_time      The current instance time.
         */
        bool PreDraw(glm::vec2 framebuffer_size, glm::vec2 window_size, double instance_time);

        /**
         * Actual drawing of Gui windows and final rendering of pushed ImGui draw commands.
         */
        bool PostDraw(void);

        /**
         * Process key events.
         */
        bool OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods);

        /**
         * Process character events.
         */
        bool OnChar(unsigned int codePoint);

        /**
         * Process mouse button events.
         */
        bool OnMouseButton(
            core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods);

        /**
         * Process mouse move events.
         */
        bool OnMouseMove(double x, double y);

        /**
         * Process mouse scroll events.
         */
        bool OnMouseScroll(double dx, double dy);

        /**
         * Triggered Shutdown.
         */
        inline bool ConsumeTriggeredShutdown(void) {
            bool request_shutdown = this->state.shutdown_triggered;
            this->state.shutdown_triggered = false;
            return request_shutdown;
        }

        /**
         * Triggered Screenshot.
         */
        bool ConsumeTriggeredScreenshot(void);
        // Valid filename is only ensured after screenshot was triggered.
        inline const std::string ConsumeScreenshotFileName(void) const {
            return this->state.screenshot_filepath;
        }

        /**
         * Set Project Script Paths.
         */
        void SetProjectScriptPaths(const std::vector<std::string>& script_paths) {
            this->state.project_script_paths = script_paths;
        }

        /**
         * Show or hide GUI
         */
        void SetVisibility(bool visible) {
            if (this->state.gui_visible != visible) {
                this->hotkeys[GUIWindows::GuiHotkeyIndex::SHOW_HIDE_GUI].is_pressed = true;
            }
        }

        /**
         * Set externally provided clipboard function and user data
         */
        void SetClipboardFunc(const char* (*get_clipboard_func)(void* user_data),
            void (*set_clipboard_func)(void* user_data, const char* string), void* user_data);

        /**
         * Synchronise changes between core graph <-> gui graph.
         *
         * - 'Old' core instance graph:    Call this function after(!) rendering of current frame.
         *                                 This way, graph changes will be applied next frame (and not 2 frames later).
         *                                 In this case in PreDraw() a gui graph is created once.
         * - 'New' megamol graph:          Call this function in GUI_Service::digestChangedRequestedResources() as
         *                                 pre-rendering step. In this case a new gui graph is created before first
         *                                 call of PreDraw() and a gui graph already exists.
         *
         * @param megamol_graph    If no megamol_graph is given, 'old' graph is synchronised via core_instance.
         */
        bool SynchronizeGraphs(megamol::core::MegaMolGraph* megamol_graph = nullptr);

    private:
        /** Available GUI styles. */
        enum Styles {
            CorporateGray,
            CorporateWhite,
            DarkColors,
            LightColors,
        };

        /** ImGui key map assignment for text manipulation hotkeys (using last unused indices < 512) */
        enum GuiTextModHotkeys { CTRL_A = 506, CTRL_C = 507, CTRL_V = 508, CTRL_X = 509, CTRL_Y = 510, CTRL_Z = 511 };

        /** The global state (for settings to be applied before ImGui::Begin). */
        struct StateBuffer {
            bool gui_visible;      // Flag indicating whether GUI is completely disabled
            bool gui_visible_post; // Required to prevent changes to 'gui_enabled' between pre and post drawing
            std::vector<WindowCollection::DrawCallbacks>
                gui_visible_buffer;   // List of all visible window IDs for restore when GUI is visible again
            bool gui_hide_next_frame; // Hiding all GUI windows properly needs two frames for ImGui to apply right state
            bool rescale_windows;     // Indicates resizing of windows for new gui zoom
            Styles style;             // Predefined GUI style
            bool style_changed;       // Flag indicating changed style
            bool autosave_gui_state;  // Automatically save state after gui has been changed
            std::vector<std::string> project_script_paths; // Project Script Path provided by Lua
            ImGuiID graph_uid;                             // UID of currently running graph
            std::vector<ImWchar> font_utf8_ranges;         // Additional UTF-8 glyph ranges for all ImGui fonts.
            bool win_save_state;    // Flag indicating that window state should be written to parameter.
            float win_save_delay;   // Flag indicating how long to wait for saving window state since last user action.
            std::string win_delete; // Name of the window to delete.
            double last_instance_time;         // Last instance time.
            bool open_popup_about;             // Flag for opening about pop-up
            bool open_popup_save;              // Flag for opening save pop-up
            bool open_popup_load;              // Flag for opening load pop-up
            bool open_popup_screenshot;        // Flag for opening screenshot file pop-up
            bool menu_visible;                 // Flag indicating menu state
            unsigned int graph_fonts_reserved; // Number of fonts reserved for the configurator graph canvas
            bool toggle_graph_entry;           // Flag indicating that the main view should be toggeled
            bool shutdown_triggered;           // Flag indicating user triggered shutdown
            bool screenshot_triggered;         // Trigger and file name for screenshot
            std::string screenshot_filepath;   // Filename the screenshot should be saved to
            int screenshot_filepath_id;        // Last unique id for screenshot filename
            std::string last_script_filename;  // Last script filename provided from lua
            bool font_apply;                   // Flag indicating whether new font should be applied
            std::string font_file_name;        // Font imgui name or font file name.
            int font_size;                     // Font size (only used whe font file name is given)
            bool hotkeys_check_once;           // WORKAROUND: Check multiple hotkey assignments once
        };

        /** The GUI hotkey array index mapping. */
        enum GuiHotkeyIndex : size_t {
            EXIT_PROGRAM = 0,
            PARAMETER_SEARCH = 1,
            SAVE_PROJECT = 2,
            LOAD_PROJECT = 3,
            MENU = 4,
            TOGGLE_GRAPH_ENTRY = 5,
            TRIGGER_SCREENSHOT = 6,
            SHOW_HIDE_GUI = 7,
            INDEX_COUNT = 8
        };

        // VARIABLES --------------------------------------------------------------

        /** Pointer to core instance. */
        megamol::core::CoreInstance* core_instance;

        /** Hotkeys */
        std::array<megamol::gui::HotkeyData_t, GuiHotkeyIndex::INDEX_COUNT> hotkeys;

        /** The ImGui context created and used by this GUIWindows */
        ImGuiContext* context;

        /** The currently initialized ImGui API */
        GUIImGuiAPI initialized_api;

        /** The window collection. */
        WindowCollection window_collection;

        /** The configurator. */
        megamol::gui::Configurator configurator;

        /** The configurator. */
        megamol::gui::LogConsole console;

        /** The current local state of the gui. */
        StateBuffer state;

        // Widgets
        FileBrowserWidget file_browser;
        StringSearchWidget search_widget;
        std::shared_ptr<TransferFunctionEditor> tf_editor_ptr;
        HoverToolTip tooltip;
        PickingBuffer picking_buffer;

        // FUNCTIONS --------------------------------------------------------------

        bool createContext(void);
        bool destroyContext(void);

        void load_default_fonts(bool reload_font_api);

        // Window Draw Callbacks
        void drawParamWindowCallback(WindowCollection::WindowConfiguration& wc);
        void drawFpsWindowCallback(WindowCollection::WindowConfiguration& wc);
        void drawTransferFunctionWindowCallback(WindowCollection::WindowConfiguration& wc);
        void drawConfiguratorWindowCallback(WindowCollection::WindowConfiguration& wc);

        void drawMenu(void);
        void drawPopUps(void);

        // Only call after ImGui::Begin() and before next ImGui::End()
        void window_sizing_and_positioning(WindowCollection::WindowConfiguration& wc, bool& out_collapsing_changed);

        bool considerModule(const std::string& modname, std::vector<std::string>& modules_list);
        void checkMultipleHotkeyAssignement(void);
        bool isHotkeyPressed(megamol::core::view::KeyCode keycode);
        void triggerCoreInstanceShutdown(void);

        std::string dump_state_to_file(const std::string& filename);
        bool load_state_from_file(const std::string& filename);

        bool state_from_json(const nlohmann::json& in_json);
        bool state_to_json(nlohmann::json& inout_json);

        void init_state(void);

        bool create_not_existing_png_filepath(std::string& inout_filepath);
    };

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUIWINDOWS_H_INCLUDED
