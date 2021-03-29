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
#include "LogConsole.h"
#include "WindowCollection.h"
#include "widgets/CorporateGreyStyle.h"
#include "widgets/CorporateWhiteStyle.h"
#include "widgets/DefaultStyle.h"
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
#include "mmcore/utility/graphics/ScreenShotComments.h"
#include "mmcore/versioninfo.h"

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

        // FUNCTIONS used in GUI_Service //////////////////////////////////////

        //////////GET //////////

        /**
         * Pass current GUI state.
         */
        std::string GetState(void) {
            return this->project_to_lua_string();
        }

        /**
         * Pass triggered Shutdown.
         */
        inline bool ConsumeTriggeredShutdown(void) {
            bool request_shutdown = this->state.shutdown_triggered;
            this->state.shutdown_triggered = false;
            return request_shutdown;
        }

        /**
         * Pass triggered Screenshot.
         */
        bool GetTriggeredScreenshot(void);

        // Valid filename is only ensured after screenshot was triggered.
        inline const std::string GetScreenshotFileName(void) const {
            return this->state.screenshot_filepath;
        }

        /**
         * Pass project load request.
         * Request is consumed when calling this function.
         */
        std::string ConsumeProjectLoadRequest(void) {
            auto project_file_name = this->state.request_load_projet_file;
            this->state.request_load_projet_file.clear();
            return project_file_name;
        }

        ///////// SET ///////////

        /**
         * Set GUI state.
         */
        void SetState(const std::string& json_state) {
            this->state.new_gui_state = json_state;
        }

        /**
         * Set GUI visibility.
         */
        void SetVisibility(bool visible) {
            // In order to take immediate effect, the GUI visibility directly is set (and not indirectly via hotkey)
            this->state.gui_visible = visible;
        }

        /**
         * Set GUI scale.
         */
        void SetScale(float scale);

        std::vector<std::tuple<std::string /*name*/,unsigned int /*GL handle*/, unsigned int /*widht*/, unsigned int /*height*/>> m_textures_test;
        void SetEntryPointTextures(std::vector<std::tuple<std::string, unsigned int, unsigned int, unsigned int>> textures) { m_textures_test = textures; }
        void ShowTextures();

        std::function<void(unsigned int, std::string const&)>* m_headnode_remote_control = nullptr;
        void SetRemoteControlCallback(std::function<void(unsigned int, std::string const&)> const& remote_control) {
            m_headnode_remote_control = &const_cast<std::function<void(unsigned int, std::string const&)>&>(remote_control);
        }
        void ShowHeadnodeRemoteControl();

        // --------------------------------------

        /**
         * Set project script paths.
         */
        void SetProjectScriptPaths(const std::vector<std::string>& script_paths) {
            this->state.project_script_paths = script_paths;
        }

        /**
         * Set current frame statistics.
         */
        void SetFrameStatistics(double last_averaged_fps, double last_averaged_ms, size_t frame_count) {
            this->state.stat_averaged_fps = last_averaged_fps;
            this->state.stat_averaged_ms = last_averaged_ms;
            this->state.stat_frame_count = frame_count;
        }

        /**
         * Set resource directories.
         */
        void SetResourceDirectories(const std::vector<std::string>& resource_directories) {
            this->state.resource_directories = resource_directories;
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
         * @param megamol_graph            If no megamol_graph is given, 'old' graph is synchronised via
         * core_instance.
         */
        bool SynchronizeGraphs(megamol::core::MegaMolGraph* megamol_graph = nullptr);

        ///////////////////////////////////////////////////////////////////////

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
                gui_visible_buffer; // List of all visible window IDs for restore when GUI is visible again
            unsigned int
                gui_hide_next_frame; // Hiding all GUI windows properly needs two frames for ImGui to apply right state
            bool rescale_windows;    // Indicates resizing of windows for new gui zoom
            Styles style;            // Predefined GUI style
            bool style_changed;      // Flag indicating changed style
            std::string new_gui_state; // If set, new gui state is applied in next graph synchronisation step
            std::vector<std::string> project_script_paths; // Project Script Path provided by Lua
            ImGuiID graph_uid;                             // UID of currently running graph
            std::vector<ImWchar> font_utf8_ranges;         // Additional UTF-8 glyph ranges for all ImGui fonts.
            bool load_fonts;                               // Flag indicating font loading
            std::string win_delete;                        // Name of the window to delete.
            double last_instance_time;                     // Last instance time.
            bool open_popup_about;                         // Flag for opening about pop-up
            bool open_popup_save;                          // Flag for opening save pop-up
            bool open_popup_load;                          // Flag for opening load pop-up
            bool open_popup_screenshot;                    // Flag for opening screenshot file pop-up
            bool menu_visible;                             // Flag indicating menu state
            unsigned int graph_fonts_reserved;             // Number of fonts reserved for the configurator graph canvas
            bool toggle_graph_entry;                       // Flag indicating that the main view should be toggeled
            bool shutdown_triggered;                       // Flag indicating user triggered shutdown
            bool screenshot_triggered;                     // Trigger and file name for screenshot
            std::string screenshot_filepath;               // Filename the screenshot should be saved to
            int screenshot_filepath_id;                    // Last unique id for screenshot filename
            bool font_apply;                               // Flag indicating whether new font should be applied
            std::string font_file_name;                    // Font imgui name or font file name.
            int font_size;                                 // Font size (only used whe font file name is given)
            std::string request_load_projet_file; // Project file name which should be loaded by fronted service
            double stat_averaged_fps;             // current average fps value
            double stat_averaged_ms;              // current average fps value
            size_t stat_frame_count;              // current fame count
            std::vector<std::string> resource_directories; // the global resource directories
            bool hotkeys_check_once;                       // WORKAROUND: Check multiple hotkey assignments once
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

        void load_default_fonts(void);

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

        std::string project_to_lua_string(void);
        bool state_from_string(const std::string& state);
        bool state_to_string(std::string& out_state);

        void init_state(void);

        void update_frame_statistics(WindowCollection::WindowConfiguration& wc);

        bool create_not_existing_png_filepath(std::string& inout_filepath);
    };

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUIWINDOWS_H_INCLUDED
