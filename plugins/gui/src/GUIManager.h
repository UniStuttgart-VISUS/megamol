/*
 * GUIManager.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GUIMANAGER_H_INCLUDED
#define MEGAMOL_GUI_GUIMANAGER_H_INCLUDED
#pragma once


#include "mmcore/CoreInstance.h"
#include "mmcore/MegaMolGraph.h"
#include "mmcore/utility/Picking_gl.h"
#include "widgets/FileBrowserWidget.h"
#include "widgets/HoverToolTip.h"
#include "widgets/PopUps.h"
#include "windows/Configurator.h"
#include "windows/WindowCollection.h"


namespace megamol {
namespace gui {


    /** ************************************************************************
     * The central class managing all GUI related stuff
     */
    class GUIManager {
    public:
        /**
         * CTOR.
         */
        GUIManager();

        /**
         * DTOR.
         */
        virtual ~GUIManager();

        /**
         * Create ImGui context using OpenGL.
         *
         * @param core_instance     The currently available core instance.
         */
        bool CreateContext(GUIImGuiAPI imgui_api);

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
        bool PostDraw();

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
         *
         * @param as_lua   If true, GUI state, scale and visibility are returned wrapped into respective LUA commands.
         *                 If false, only GUI state JSON string is returned.
         */
        inline std::string GetState(bool as_lua) {
            return this->project_to_lua_string(as_lua);
        }

        /**
         * Pass current GUI visibility.
         */
        inline bool GetVisibility() const {
            return this->gui_state.gui_visible;
        }

        /**
         * Pass current GUI scale.
         */
        float GetScale() const {
            return megamol::gui::gui_scaling.Get();
        }

        /**
         * Pass triggered Shutdown.
         */
        inline bool GetTriggeredShutdown() {
            bool request_shutdown = this->gui_state.shutdown_triggered;
            this->gui_state.shutdown_triggered = false;
            return request_shutdown;
        }

        /**
         * Pass triggered Screenshot.
         */
        bool GetTriggeredScreenshot() {
            bool trigger_screenshot = this->gui_state.screenshot_triggered;
            this->gui_state.screenshot_triggered = false;
            return trigger_screenshot;
        }

        /**
         * Return current screenshot file name and create new file name for next screenshot
         */
        inline std::string GetScreenshotFileName() {
            auto screenshot_filepath = this->gui_state.screenshot_filepath;
            this->create_unique_screenshot_filename(this->gui_state.screenshot_filepath);
            return screenshot_filepath;
        }

        /**
         * Pass project load request.
         * Request is consumed when calling this function.
         */
        std::string GetProjectLoadRequest() {
            auto project_file_name = this->gui_state.request_load_projet_file;
            this->gui_state.request_load_projet_file.clear();
            return project_file_name;
        }

        /**
         * Pass current mouse cursor request.
         *
         * See imgui.h: enum ImGuiMouseCursor_
         * io.MouseDrawCursor is true when Software Cursor should be drawn instead.
         *
         * @return Retured mouse cursor is in range [ImGuiMouseCursor_None=-1, (ImGuiMouseCursor_COUNT-1)=8]
         */
        int GetMouseCursor() const {
            return ((!ImGui::GetIO().MouseDrawCursor) ? (ImGui::GetMouseCursor()) : (ImGuiMouseCursor_None));
        }

        ///////// SET ///////////

        /**
         * Set GUI state.
         */
        void SetState(const std::string& json_state) {
            this->gui_state.new_gui_state = json_state;
        }

        /**
         * Set GUI visibility.
         */
        void SetVisibility(bool visible) {
            // In order to take immediate effect, the GUI visibility directly is set (and not indirectly via hotkey)
            this->gui_state.gui_visible = visible;
        }

        /**
         * Set GUI scale.
         */
        void SetScale(float scale);

        /**
         * Set project script paths.
         */
        void SetProjectScriptPaths(const std::vector<std::string>& script_paths) {
            this->gui_state.project_script_paths = script_paths;
        }

        /**
         * Set current frame statistics.
         */
        void SetFrameStatistics(double last_averaged_fps, double last_averaged_ms, size_t frame_count) {
            this->gui_state.stat_averaged_fps = static_cast<float>(last_averaged_fps);
            this->gui_state.stat_averaged_ms = static_cast<float>(last_averaged_ms);
            this->gui_state.stat_frame_count = frame_count;
        }

        /**
         * Set resource directories.
         */
        void SetResourceDirectories(const std::vector<std::string>& resource_directories) {
            megamol::gui::gui_resource_paths = resource_directories;
        }

        /**
         * Set externally provided clipboard function and user data
         */
        void SetClipboardFunc(const char* (*get_clipboard_func)(void* user_data),
            void (*set_clipboard_func)(void* user_data, const char* string), void* user_data);

        /**
         * Register GUI window/pop-up/notification for external window callback function containing ImGui content.
         *
         * @param window_name   The unique window name
         * @param callback      The window callback function containing the GUI content of the window
         */
        void RegisterWindow(
            const std::string& window_name, std::function<void(AbstractWindow::BasicConfig&)> const& callback);

        void RegisterPopUp(const std::string& name, bool& open, std::function<void()> const& callback);

        void RegisterNotification(const std::string& name, bool& open, const std::string& message);

        /**
         * Synchronise changes between core graph <-> gui graph.
         *
         * @param megamol_graph          The megamol graph.
         * @param core_instance          The core_instance.
         */
        bool SynchronizeRunningGraph(
            megamol::core::MegaMolGraph& megamol_graph, megamol::core::CoreInstance& core_instance);

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
            std::vector<std::string>
                gui_restore_hidden_windows; // List of all visible window IDs for restore when GUI is visible again
            unsigned int
                gui_hide_next_frame; // Hiding all GUI windows properly needs two frames for ImGui to apply right state
            Styles style;            // Predefined GUI style
            bool style_changed;      // Flag indicating changed style
            std::string new_gui_state; // If set, new gui state is applied in next graph synchronisation step
            std::vector<std::string> project_script_paths; // Project Script Path provided by Lua
            std::vector<ImWchar> font_utf8_ranges;         // Additional UTF-8 glyph ranges for all ImGui fonts.
            bool load_default_fonts;                       // Flag indicating font loading
            size_t win_delete_hash_id;                     // Hash id of the window to delete.
            double last_instance_time;                     // Last instance time.
            bool open_popup_about;                         // Flag for opening about pop-up
            bool open_popup_save;                          // Flag for opening save pop-up
            bool open_popup_load;                          // Flag for opening load pop-up
            bool open_popup_screenshot;                    // Flag for opening screenshot file pop-up
            bool menu_visible;                             // Flag indicating menu state
            unsigned int graph_fonts_reserved;             // Number of fonts reserved for the configurator graph canvas
            bool shutdown_triggered;                       // Flag indicating user triggered shutdown
            bool screenshot_triggered;                     // Trigger and file name for screenshot
            std::string screenshot_filepath;               // Filename the screenshot should be saved to
            int screenshot_filepath_id;                    // Last unique id for screenshot filename
            unsigned int font_load;               // Flag indicating whether new font should be applied in which frame
            std::string font_load_filename;       // Font imgui name or font file name.
            int font_load_size;                   // Font size (only used whe font file name is given)
            std::string request_load_projet_file; // Project file name which should be loaded by fronted service
            float stat_averaged_fps;              // current average fps value
            float stat_averaged_ms;               // current average fps value
            size_t stat_frame_count;              // current fame count
            bool load_docking_preset;             // Flag indicating docking preset loading
        };


        // VARIABLES --------------------------------------------------------------

        megamol::gui::HotkeyMap_t hotkeys;

        /** The ImGui context created and used by this GUIManager */
        ImGuiContext* context;

        GUIImGuiAPI initialized_api;

        /** The current local state of the gui. */
        StateBuffer gui_state;

        /** GUI element collections. */
        WindowCollection win_collection;
        std::map<std::string, std::pair<bool*, std::function<void()>>> popup_collection;
        std::map<std::string, std::tuple<bool*, bool, std::string>> notification_collection;

        /** Shortcut pointer to configurator window */
        std::shared_ptr<Configurator> win_configurator_ptr;

        // Widgets
        FileBrowserWidget file_browser;
        HoverToolTip tooltip;
        megamol::core::utility::PickingBuffer picking_buffer;

        // FUNCTIONS --------------------------------------------------------------

        void init_state();

        bool create_context();

        bool destroy_context();

        void load_default_fonts();

        void draw_menu();

        void draw_popups();

        bool is_hotkey_pressed(megamol::core::view::KeyCode keycode) const;

        void load_preset_window_docking(ImGuiID global_docking_id);

        void load_imgui_settings_from_string(const std::string& imgui_settings);

        std::string save_imgui_settings_to_string() const;

        std::string project_to_lua_string(bool as_lua);

        bool state_from_string(const std::string& state);

        bool state_to_string(std::string& out_state);

        bool create_unique_screenshot_filename(std::string& inout_filepath);

        std::string extract_fontname(const std::string& imgui_fontname) const;
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUIMANAGER_H_INCLUDED
