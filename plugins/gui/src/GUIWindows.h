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
#include "FileUtils.h"
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
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/versioninfo.h"
#include "mmcore/view/AbstractView_EventConsumption.h"

#include "vislib/math/Rectangle.h"

#include <ctime>
#include <iomanip>
#include <sstream>

// Used for platform independent clipboard (ImGui so far only provides windows implementation)
#ifdef GUI_USE_GLFW
#include "GLFW/glfw3.h"
#endif


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
        bool CreateContext_GL(megamol::core::CoreInstance* core_instance);

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
         * Return list of parameter slots provided by this class. Make available in module which uses this class.
         */
        inline const std::vector<megamol::core::param::ParamSlot*> GetParams(void) const {
            return this->param_slots;
        }

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
            this->project_script_paths = script_paths;
        }

        /**
         * Synchronise changes between core graph <-> gui graph.
         *
         * - 'Old' core instance graph:    Call this function after(!) rendering of current frame.
         *                                 This way, graph changes will be applied next frame (and not 2 frames later).
         *                                 In this case in PreDraw() a gui graph is created once.
         * - 'New' megamol graph:          Call this function in GUI_Service::digestChangedRequestedResources() as
         * pre-rendering step. In this case a new gui graph is created before first call of PreDraw() and a gui graph
         * already exists.
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
            ImGuiID graph_uid;                     // UID of currently running graph
            std::string font_file;                 // Apply changed font file name.
            float font_size;                       // Apply changed font size.
            unsigned int font_index;               // Apply cahnged font by index.
            std::vector<ImWchar> font_utf8_ranges; // Additional UTF-8 glyph ranges for all ImGui fonts.
            bool win_save_state;                   // Flag indicating that window state should be written to parameter.
            float win_save_delay;   // Flag indicating how long to wait for saving window state since last user action.
            std::string win_delete; // Name of the window to delete.
            double last_instance_time;         // Last instance time.
            bool open_popup_about;             // Flag for opening about pop-up
            bool open_popup_save;              // Flag for opening save pop-up
            bool open_popup_load;              // Flag for opening load pop-up
            bool open_popup_screenshot;        // Flag for opening screenshot file pop-up
            bool menu_visible;                 // Flag indicating menu state
            unsigned int graph_fonts_reserved; // Number of fonts reserved for the configurator graph canvas
            bool toggle_main_view;             // Flag indicating that the main view should be toggeled
            bool shutdown_triggered;           // Flag indicating user triggered shutdown
            bool screenshot_triggered;         // Trigger and file name for screenshot
            std::string screenshot_filepath;   // Filename the screenshot should be saved to
            int screenshot_filepath_id;        // Last unique id for screenshot filename
            bool hotkeys_check_once;           // WORKAROUND: Check multiple hotkey assignments once
        };

        /** The GUI hotkey array index mapping. */
        enum GuiHotkeyIndex : size_t {
            EXIT_PROGRAM = 0,
            PARAMETER_SEARCH = 1,
            SAVE_PROJECT = 2,
            LOAD_PROJECT = 3,
            MENU = 4,
            TOGGLE_MAIN_VIEWS = 5,
            TRIGGER_SCREENSHOT = 6,
            RESET_WINDOWS_POS = 7,
            INDEX_COUNT = 8
        };

        // VARIABLES --------------------------------------------------------------

        /** Pointer to core instance. */
        megamol::core::CoreInstance* core_instance;

        /** List of pointers to all paramters. */
        std::vector<megamol::core::param::ParamSlot*> param_slots;

        /** A parameter to select the style */
        megamol::core::param::ParamSlot style_param;
        /** A parameter to store the profile */
        megamol::core::param::ParamSlot state_param;
        /** A parameter for automatically saving gui state to file */
        megamol::core::param::ParamSlot autosave_state_param;
        /** A parameter for automatically start the configurator at start up */
        megamol::core::param::ParamSlot autostart_configurator_param;

        /** Hotkeys */
        std::array<megamol::gui::HotkeyData_t, GuiHotkeyIndex::INDEX_COUNT> hotkeys;

        /** The ImGui context created and used by this GUIWindows */
        ImGuiContext* context;

        /** The currently initialized ImGui API */
        GUIImGuiAPI api;

        /** The window collection. */
        WindowCollection window_collection;

        /** The configurator. */
        megamol::gui::Configurator configurator;

        /** The current local state of the gui. */
        StateBuffer state;

        /** Project Script Path provided by Lua */
        std::vector<std::string> project_script_paths;

        // Widgets
        FileBrowserWidget file_browser;
        StringSearchWidget search_widget;
        std::shared_ptr<TransferFunctionEditor> tf_editor_ptr;
        HoverToolTip tooltip;
        PickingBuffer picking_buffer;
        PickableTriangle triangle_widget;

        // FUNCTIONS --------------------------------------------------------------

        bool createContext(void);
        bool destroyContext(void);

        void validateParameters();

        // Window Draw Callbacks
        void drawParamWindowCallback(WindowCollection::WindowConfiguration& wc);
        void drawFpsWindowCallback(WindowCollection::WindowConfiguration& wc);
        void drawFontWindowCallback(WindowCollection::WindowConfiguration& wc);
        void drawTransferFunctionWindowCallback(WindowCollection::WindowConfiguration& wc);
        void drawConfiguratorWindowCallback(WindowCollection::WindowConfiguration& wc);

        void drawMenu(void);
        void drawPopUps(void);

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
