/**
 * MegaMol
 * Copyright (c) 2018, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <implot.h>

#include "CommandRegistry.h"
#include "FrontendResource.h"
#include "PluginsResource.h"
#include "gui_render_backend.h"
#include "mmcore/MegaMolGraph.h"
#include "mmcore/utility/Picking.h"
#include "widgets/FileBrowserWidget.h"
#include "widgets/HoverToolTip.h"
#include "widgets/PopUps.h"
#include "windows/AnimationEditor.h"
#include "windows/Configurator.h"
#include "windows/LogConsole.h"
#include "windows/WindowCollection.h"


namespace megamol::gui {


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
     * @param backend     The ImGui render backend to use.
     */
    bool CreateContext(GUIRenderBackend backend);

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
    inline float GetScale() const {
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
    inline bool GetTriggeredScreenshot() {
        bool trigger_screenshot = this->gui_state.screenshot_triggered;
        this->gui_state.screenshot_triggered = false;
        return trigger_screenshot;
    }

    /**
     * Return current screenshot file name and create new file name for next screenshot
     */
    inline std::filesystem::path GetScreenshotFileName() {
        auto screenshot_filepath = std::filesystem::u8path(this->gui_state.screenshot_filepath);
        this->create_unique_screenshot_filename(this->gui_state.screenshot_filepath);
        return screenshot_filepath;
    }

    /**
     * Pass project load request.
     * Request is consumed when calling this function.
     */
    inline std::filesystem::path GetProjectLoadRequest() {
        auto project_file_name = std::filesystem::u8path(this->gui_state.request_load_projet_file);
        this->gui_state.request_load_projet_file.clear();
        return project_file_name;
    }

    /**
     * Pass current mouse cursor request.
     *
     * See imgui.h: enum ImGuiMouseCursor_
     * io.MouseDrawCursor is true when Software Cursor should be drawn instead.
     *
     * @return Returned mouse cursor is in range [ImGuiMouseCursor_None=-1, (ImGuiMouseCursor_COUNT-1)=8]
     */
    inline int GetMouseCursor() const {
        return ((!ImGui::GetIO().MouseDrawCursor) ? (ImGui::GetMouseCursor()) : (ImGuiMouseCursor_None));
    }

    /**
     * Get image data containing rendered image.
     */
    inline megamol::frontend_resources::ImageWrapper GetImage() {
        return this->render_backend.GetImage();
    }

    /**
     * Get the updated animation editor data.
     */
    frontend_resources::AnimationEditorData& GetAnimationEditorData() {
        return this->win_animation_editor_ptr->GetAnimationEditorData();
    }


    ///////// SET ///////////

    /**
     * Set GUI state.
     */
    inline void SetState(const std::string& json_state) {
        this->gui_state.new_gui_state = json_state;
    }

    /**
     * Set GUI visibility.
     */
    void SetVisibility(bool visible) {
        this->gui_state.gui_visible = visible;
    }

    /**
     * Set GUI scale.
     */
    void SetScale(float scale);

    /**
     * Set project script paths.
     */
    inline void SetProjectScriptPaths(const std::vector<std::string>& script_paths) {
        this->gui_state.project_script_paths = script_paths;
    }

    /**
     * Set current frame statistics.
     */
    inline void SetFrameStatistics(
        double last_averaged_fps, double last_averaged_ms, size_t frame_count, double last_frame_millis) {
        this->gui_state.stat_averaged_fps = static_cast<float>(last_averaged_fps);
        this->gui_state.stat_averaged_ms = static_cast<float>(last_averaged_ms);
        this->gui_state.stat_frame_count = frame_count;
        this->gui_state.last_frame_ms = static_cast<float>(last_frame_millis);
    }

    /**
     * Set resource directories.
     */
    inline void SetResourceDirectories(const std::vector<std::string>& resource_directories) {
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

    void RegisterPopUp(const std::string& name, std::weak_ptr<bool> open, std::function<void()> const& callback);

    void RegisterNotification(const std::string& name, std::weak_ptr<bool> open, const std::string& message);

    bool InitializeGraphSynchronisation(const megamol::frontend_resources::PluginsResource& pluginsRes) {
        return this->win_configurator_ptr->GetGraphCollection().InitializeGraphSynchronisation(pluginsRes);
    }

    bool SynchronizeGraphs(megamol::core::MegaMolGraph& megamol_graph);

    /**
     * Register GUI hotkeys.
     */
    void RegisterHotkeys(megamol::core::view::CommandRegistry& cmdregistry, megamol::core::MegaMolGraph& megamol_graph);

    void SetLuaFunc(megamol::frontend_resources::common_types::lua_func_type* lua_func) {
        auto cons = win_collection.GetWindow<LogConsole>();
        if (cons) {
            cons->SetLuaFunc(lua_func);
        }
        this->win_configurator_ptr->GetGraphCollection().SetLuaFunc(lua_func);
        this->win_animation_editor_ptr->SetLuaFunc(lua_func);
    }

#ifdef MEGAMOL_USE_PROFILING
    void SetPerformanceManager(frontend_resources::performance::PerformanceManager* perf_manager) {
        this->win_configurator_ptr->GetGraphCollection().SetPerformanceManager(perf_manager);
    }

    frontend_resources::performance::ProfilingLoggingStatus* perf_logging;

    void SetProfilingLoggingStatus(frontend_resources::performance::ProfilingLoggingStatus* perf_logging_status) {
        this->perf_logging = perf_logging_status;
    }
    void AppendPerformanceData(const frontend_resources::performance::frame_info& fi) {
        this->win_configurator_ptr->GetGraphCollection().AppendPerformanceData(fi);
    }
#endif

    bool NotifyRunningGraph_AddModule(core::ModuleInstance_t const& module_inst) {
        return this->win_configurator_ptr->GetGraphCollection().NotifyRunningGraph_AddModule(module_inst);
    }
    bool NotifyRunningGraph_DeleteModule(core::ModuleInstance_t const& module_inst) {
        return this->win_configurator_ptr->GetGraphCollection().NotifyRunningGraph_DeleteModule(module_inst);
    }
    bool NotifyRunningGraph_RenameModule(
        std::string const& old_name, std::string const& new_name, core::ModuleInstance_t const& module_inst) {
        return this->win_configurator_ptr->GetGraphCollection().NotifyRunningGraph_RenameModule(
            old_name, new_name, module_inst);
    }
    bool NotifyRunningGraph_AddParameters(
        std::vector<megamol::frontend_resources::ModuleGraphSubscription::ParamSlotPtr> const& param_slots) {
        return this->win_configurator_ptr->GetGraphCollection().NotifyRunningGraph_AddParameters(param_slots);
    }
    bool NotifyRunningGraph_RemoveParameters(
        std::vector<megamol::frontend_resources::ModuleGraphSubscription::ParamSlotPtr> const& param_slots) {
        return this->win_configurator_ptr->GetGraphCollection().NotifyRunningGraph_RemoveParameters(param_slots);
    }
    bool NotifyRunningGraph_ParameterChanged(
        megamol::frontend_resources::ModuleGraphSubscription::ParamSlotPtr const& param_slot,
        std::string const& new_value) {
        bool ret = this->win_animation_editor_ptr->NotifyParamChanged(param_slot, new_value);
        ret |=
            this->win_configurator_ptr->GetGraphCollection().NotifyRunningGraph_ParameterChanged(param_slot, new_value);
        return ret;
    }
    bool NotifyRunningGraph_AddCall(core::CallInstance_t const& call_inst) {
        return this->win_configurator_ptr->GetGraphCollection().NotifyRunningGraph_AddCall(call_inst);
    }
    bool NotifyRunningGraph_DeleteCall(core::CallInstance_t const& call_inst) {
        return this->win_configurator_ptr->GetGraphCollection().NotifyRunningGraph_DeleteCall(call_inst);
    }
    bool NotifyRunningGraph_EnableEntryPoint(core::ModuleInstance_t const& module_inst) {
        return this->win_configurator_ptr->GetGraphCollection().NotifyRunningGraph_EnableEntryPoint(module_inst);
    }
    bool NotifyRunningGraph_DisableEntryPoint(core::ModuleInstance_t const& module_inst) {
        return this->win_configurator_ptr->GetGraphCollection().NotifyRunningGraph_DisableEntryPoint(module_inst);
    }

    std::vector<std::string> requested_lifetime_resources() const {
        return requested_resources;
    }

    void setRequestedResources(std::shared_ptr<frontend_resources::FrontendResourcesMap> const& resources);

    ///////////////////////////////////////////////////////////////////////

private:
    /** Available GUI styles. */
    enum Styles {
        CorporateGray,
        CorporateWhite,
        DarkColors,
        LightColors,
    };

    /** The global state (for settings to be applied before ImGui::Begin). */
    struct StateBuffer {
        bool gui_visible;      // Flag indicating whether GUI is completely disabled
        bool gui_visible_post; // Required to prevent changes to 'gui_enabled' between pre and post drawing
        std::vector<std::string>
            gui_restore_hidden_windows; // List of all visible window IDs for restore when GUI is visible again
        unsigned int
            gui_hide_next_frame;   // Hiding all GUI windows properly needs two frames for ImGui to apply right state
        Styles style;              // Predefined GUI style
        bool style_changed;        // Flag indicating changed style
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
        bool open_popup_font;                          // Flag for opening font pop-up
        bool menu_visible;                             // Flag indicating menu state
        bool show_imgui_metrics;                       // Flag indicating ImGui metrics window
        unsigned int graph_fonts_reserved;             // Number of fonts reserved for the configurator graph canvas
        bool shutdown_triggered;                       // Flag indicating user triggered shutdown
        bool screenshot_triggered;                     // Trigger and file name for screenshot
        std::string screenshot_filepath;               // Filename the screenshot should be saved to
        int screenshot_filepath_id;                    // Last unique id for screenshot filename
        unsigned int font_load;               // Flag indicating whether new font should be applied in which frame
        std::string font_load_name;           // Font imgui name or font file name.
        std::string font_input_string_buffer; // Widget string buffer for font input.
        std::string
            default_font_filename; // File name of the currently loaded font (only set when new font is loaded from file).
        int font_load_size;        // Font size (only used whe font file name is given)
        std::string request_load_projet_file; // Project file name which should be loaded by fronted service
        float stat_averaged_fps;              // current average fps value
        float stat_averaged_ms;               // current average fps value
        size_t stat_frame_count;              // current frame count
        float last_frame_ms;                  // last frame time
        bool load_docking_preset;             // Flag indicating docking preset loading
        float window_alpha;                   // Global transparency value for window background
        float scale_input_float_buffer;       // Widget float buffer for scale input
    };

    // VARIABLES --------------------------------------------------------------

    megamol::gui::HotkeyMap_t gui_hotkeys;

    /** The ImGui context */
    ImGuiContext* imgui_context;

    /** The ImGui render backend  */
    gui_render_backend render_backend;

    /** The ImPlot context  */
    ImPlotContext* implot_context;

    /** The current local state of the gui. */
    StateBuffer gui_state;

    /** GUI element collections. */
    WindowCollection win_collection;

    /** Resource requests from the GUI windows. */
    std::vector<std::string> requested_resources;

    struct PopUpData {
        std::weak_ptr<bool> open_flag;
        std::function<void()> draw_callback;
    };
    std::map<std::string, PopUpData> popup_collection;

    struct NotificationData {
        std::weak_ptr<bool> open_flag;
        bool disable;
        std::string message;
    };
    std::map<std::string, NotificationData> notification_collection;

    /** Shortcut pointer to configurator window */
    std::shared_ptr<Configurator> win_configurator_ptr;
    std::shared_ptr<AnimationEditor> win_animation_editor_ptr;

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

    void load_preset_window_docking(ImGuiID global_docking_id);

    void load_imgui_settings_from_string(const std::string& imgui_settings);

    std::string save_imgui_settings_to_string() const;

    std::string project_to_lua_string(bool as_lua);

    bool state_from_string(const std::string& state);

    bool state_to_string(std::string& out_state);

    bool create_unique_screenshot_filename(std::string& inout_filepath);

    std::string extract_fontname(const std::string& imgui_fontname) const;
};


} // namespace megamol::gui
