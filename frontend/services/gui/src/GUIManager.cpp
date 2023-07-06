/*
 * GUIManager.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "GUIManager.h"
#include "imgui_stdlib.h"
#include "mmcore/utility/FileUtils.h"
#include "mmcore/utility/buildinfo/BuildInfo.h"
#include "widgets/ButtonWidgets.h"
#include "widgets/CorporateGreyStyle.h"
#include "widgets/CorporateWhiteStyle.h"
#include "widgets/DefaultStyle.h"
#include "windows/HotkeyEditor.h"
#include "windows/PerformanceMonitor.h"


using namespace megamol::gui;


GUIManager::GUIManager()
        : gui_hotkeys()
        , imgui_context(nullptr)
        , render_backend()
        , implot_context(nullptr)
        , gui_state()
        , win_collection()
        , popup_collection()
        , notification_collection()
        , win_animation_editor_ptr(nullptr)
        , win_configurator_ptr(nullptr)
        , file_browser()
        , tooltip()
        , picking_buffer() {

    // Init hotkeys
    this->gui_hotkeys[HOTKEY_GUI_TRIGGER_SCREENSHOT] = {"_hotkey_gui_trigger_screenshot",
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F2, core::view::Modifier::NONE), false};
    this->gui_hotkeys[HOTKEY_GUI_TOGGLE_GRAPH_ENTRY] = {"_hotkey_gui_toggle_graph_entry",
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F3, core::view::Modifier::NONE), false};
    this->gui_hotkeys[HOTKEY_GUI_EXIT_PROGRAM] = {"_hotkey_gui_exit",
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F4, core::view::Modifier::ALT), false};
    this->gui_hotkeys[HOTKEY_GUI_MENU] = {"_hotkey_gui_menu",
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F12, core::view::Modifier::NONE), false};
    this->gui_hotkeys[HOTKEY_GUI_SAVE_PROJECT] = {"_hotkey_gui_save_project",
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_S, core::view::Modifier::CTRL), false};
    this->gui_hotkeys[HOTKEY_GUI_LOAD_PROJECT] = {"_hotkey_gui_load_project",
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_L, core::view::Modifier::CTRL), false};
    this->gui_hotkeys[HOTKEY_GUI_SHOW_HIDE_GUI] = {"_hotkey_gui_show-hide",
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_G, core::view::Modifier::CTRL), false};

    this->win_configurator_ptr = this->win_collection.GetWindow<Configurator>();
    this->win_animation_editor_ptr = this->win_collection.GetWindow<AnimationEditor>();
    assert(this->win_configurator_ptr != nullptr);
    assert(this->win_animation_editor_ptr != nullptr);

    requested_resources = win_collection.requested_lifetime_resources();
#ifdef MEGAMOL_USE_PROFILING
    requested_resources.push_back(frontend_resources::Performance_Logging_Status_Req_Name);
#endif

    this->init_state();
}


GUIManager::~GUIManager() {

    this->destroy_context();
}


void megamol::gui::GUIManager::init_state() {

    this->gui_state.gui_visible = true;
    this->gui_state.gui_visible_post = true;
    this->gui_state.style = GUIManager::Styles::DarkColors;
    this->gui_state.style_changed = true;
    this->gui_state.new_gui_state = "";
    this->gui_state.project_script_paths.clear();
    this->gui_state.font_utf8_ranges.clear();
    this->gui_state.load_default_fonts = false;
    this->gui_state.win_delete_hash_id = 0;
    this->gui_state.last_instance_time = 0.0;
    this->gui_state.open_popup_about = false;
    this->gui_state.open_popup_save = false;
    this->gui_state.open_popup_load = false;
    this->gui_state.open_popup_screenshot = false;
    this->gui_state.open_popup_font = false;
    this->gui_state.menu_visible = true;
    this->gui_state.show_imgui_metrics = false;
    this->gui_state.graph_fonts_reserved = 0;
    this->gui_state.shutdown_triggered = false;
    this->gui_state.screenshot_triggered = false;
    this->gui_state.screenshot_filepath = "megamol_screenshot.png";
    this->create_unique_screenshot_filename(this->gui_state.screenshot_filepath);
    this->gui_state.screenshot_filepath_id = 0;
    this->gui_state.font_load = 0;
    this->gui_state.font_load_name = "";
    this->gui_state.font_load_size = 12;
    this->gui_state.font_input_string_buffer = "";
    this->gui_state.default_font_filename = "";
    this->gui_state.request_load_projet_file = "";
    this->gui_state.stat_averaged_fps = 0.0f;
    this->gui_state.stat_averaged_ms = 0.0f;
    this->gui_state.stat_frame_count = 0;
    this->gui_state.load_docking_preset = true;
    this->gui_state.window_alpha = 1.0f;
    this->gui_state.scale_input_float_buffer = 1.0f;
}


bool GUIManager::CreateContext(GUIRenderBackend backend) {

    if (!this->render_backend.CheckPrerequisites(backend)) {
        return false;
    }
    bool previous_imgui_context_exists = (ImGui::GetCurrentContext() != nullptr);
    if (this->create_context()) {
        // If previous imgui context has been created, render backend already has been initialized.
        if (!previous_imgui_context_exists) {
            if (!this->render_backend.Init(backend)) {
                this->destroy_context();
                return false;
            }
        }
        megamol::gui::gui_context_count++;
        ImGui::SetCurrentContext(this->imgui_context);
        return true;
    }
    return false;
}


bool GUIManager::PreDraw(glm::vec2 framebuffer_size, glm::vec2 window_size, double instance_time) {

    // Handle multiple ImGui contexts.
    bool valid_imgui_scope =
        ((ImGui::GetCurrentContext() != nullptr) ? (ImGui::GetCurrentContext()->WithinFrameScope) : (false));
    if (this->gui_state.gui_visible && valid_imgui_scope) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Nesting ImGui contexts is not supported. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (!this->render_backend.IsBackendInitialized()) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No ImGui render backend initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->imgui_context == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No valid ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Set ImGui context
    ImGui::SetCurrentContext(this->imgui_context);

    // Delayed font loading for resource directories being available via resource in frontend
    if (this->gui_state.load_default_fonts) {
        this->load_default_fonts();
        this->gui_state.load_default_fonts = false;
    }

    // Required to prevent change in gui drawing between pre and post draw
    this->gui_state.gui_visible_post = this->gui_state.gui_visible;
    // Early exit when pre step should be omitted
    if (!this->gui_state.gui_visible) {
        this->render_backend.ClearFrame();
        return true;
    }

    // Set stuff for next frame --------------------------------------------
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(window_size.x, window_size.y);
    if ((window_size.x > 0.0f) && (window_size.y > 0.0f)) {
        io.DisplayFramebufferScale = ImVec2(framebuffer_size.x / window_size.x, framebuffer_size.y / window_size.y);
    }

#ifdef GUI_VERBOSE
    if ((instance_time - this->gui_state.last_instance_time) < 0.0) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Current instance time results in negative time delta. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
    }
#endif // GUI_VERBOSE
    io.DeltaTime = ((instance_time - this->gui_state.last_instance_time) > 0.0)
                       ? (static_cast<float>(instance_time - this->gui_state.last_instance_time))
                       : (io.DeltaTime);
    this->gui_state.last_instance_time = ((instance_time - this->gui_state.last_instance_time) > 0.0)
                                             ? (instance_time)
                                             : (this->gui_state.last_instance_time + io.DeltaTime);

    // Style
    if (this->gui_state.style_changed) {
        ImGuiStyle& style = ImGui::GetStyle();
        switch (this->gui_state.style) {
        case (GUIManager::Styles::DarkColors): {
            DefaultStyle();
            ImGui::StyleColorsDark();
            style.WindowRounding = 0.0f;
            style.Colors[ImGuiCol_ChildBg] = style.Colors[ImGuiCol_WindowBg];
        } break;
        case (GUIManager::Styles::LightColors): {
            DefaultStyle();
            ImGui::StyleColorsLight();
            style.WindowRounding = 0.0f;
            style.Colors[ImGuiCol_ChildBg] = style.Colors[ImGuiCol_WindowBg];
        } break;
        case (GUIManager::Styles::CorporateGray): {
            CorporateGreyStyle();
        } break;
        case (GUIManager::Styles::CorporateWhite): {
            CorporateWhiteStyle();
        } break;
        default:
            break;
        }
        style.Colors[ImGuiCol_WindowBg].w = this->gui_state.window_alpha;
        style.Colors[ImGuiCol_ChildBg].w = this->gui_state.window_alpha;

        // Set tesselation error: Smaller value => better tesselation of circles and round corners.
        style.CircleTessellationMaxError = 0.3f;
        // Scale all ImGui style options with current scaling factor
        style.ScaleAllSizes(megamol::gui::gui_scaling.Get());

        this->gui_state.style_changed = false;
    }

    // Propagate window specific data
    if (auto win_perfmon_ptr = this->win_collection.GetWindow<PerformanceMonitor>()) {
        win_perfmon_ptr->SetData(
            this->gui_state.stat_averaged_fps, this->gui_state.stat_averaged_ms, this->gui_state.stat_frame_count);
    }
    if (win_animation_editor_ptr != nullptr) {
        win_animation_editor_ptr->SetLastFrameMillis(gui_state.last_frame_ms);
    }

    // Update windows
    this->win_collection.Update();

    // Delete window
    if (this->gui_state.win_delete_hash_id != 0) {
        this->win_collection.DeleteWindow(this->gui_state.win_delete_hash_id);
        this->gui_state.win_delete_hash_id = 0;
    }

    // Start new ImGui frame --------------------------------------------------
    this->render_backend.NewFrame(framebuffer_size, window_size);
    ImGui::NewFrame();

/// DOCKING
#ifdef IMGUI_HAS_DOCK
    // Global Docking Space --------------------------------------------------
    ImGuiStyle& style = ImGui::GetStyle();
    auto child_bg = style.Colors[ImGuiCol_ChildBg];
    style.Colors[ImGuiCol_ChildBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    auto global_docking_id =
        ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
    style.Colors[ImGuiCol_ChildBg] = child_bg;

    // Load global docking preset(before first window is drawn!)
    if (this->gui_state.load_docking_preset) {
        this->load_preset_window_docking(global_docking_id);
        this->gui_state.load_docking_preset = false;
    }
#endif

    return true;
}


bool GUIManager::PostDraw() {

    if (this->gui_state.gui_visible_post) {

        if (!this->render_backend.IsBackendInitialized()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] No ImGui render backend initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        if (this->imgui_context == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] No valid ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        // Enable ImGui context
        ImGui::SetCurrentContext(this->imgui_context);
        ImGuiIO& io = ImGui::GetIO();
        ImGuiStyle& style = ImGui::GetStyle();

        ////////// DRAW GUI ///////////////////////////////////////////////////////

        // Enable backend rendering
        auto width = static_cast<int>(io.DisplaySize.x);
        auto height = static_cast<int>(io.DisplaySize.y);
        this->render_backend.EnableRendering(width, height);

        megamol::gui::gui_mouse_wheel = io.MouseWheel;

        try {

            // Draw global menu -----------------------------------------------
            this->draw_menu();

            // Draw Windows and their pop-ups ---------------------------------
            this->win_collection.Draw(this->gui_state.menu_visible);

            // Draw global pop-ups --------------------------------------------
            this->draw_popups();

            // Draw global parameter widgets ----------------------------------
            if (auto graph_ptr = this->win_configurator_ptr->GetGraphCollection().GetRunningGraph()) {
                /// ! Only enabled in second frame if interaction objects are added during first frame !
                this->picking_buffer.EnableInteraction(glm::vec2(io.DisplaySize.x, io.DisplaySize.y));

                graph_ptr->DrawGlobalParameterWidgets(
                    this->picking_buffer, this->win_collection.GetWindow<TransferFunctionEditor>());

                this->picking_buffer.DisableInteraction();
            }

        } catch (std::exception ex) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Unknown Error... [%s at %s, %s, line %d]\n", ex.what(), __FILE__, __FUNCTION__, __LINE__);
        }

        // Render the current ImGui frame
        ImGui::Render();
        auto draw_data = ImGui::GetDrawData();
        // Backend rendering
        this->render_backend.Render(draw_data);

        ///////////////////////////////////////////////////////////////////////////

        // Loading new font -------------------------------------------------------
        // (after first imgui frame for default fonts being available)
        if (this->gui_state.font_load > 1) {
            this->gui_state.font_load--;
        } else if (this->gui_state.font_load == 1) {
            bool load_success = false;

            if (!this->render_backend.SupportsCustomFonts()) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[GUI] Ignoring loading of custom font. Unsupported feature by currently used render backend.");

            } else if (megamol::core::utility::FileUtils::FileWithExtensionExists<std::string>(
                           this->gui_state.font_load_name, std::string("ttf"))) {

                ImFontConfig config;
                config.OversampleH = 4;
                config.OversampleV = 4;
                config.GlyphRanges = this->gui_state.font_utf8_ranges.data();

                if (io.Fonts->AddFontFromFileTTF(this->gui_state.font_load_name.c_str(),
                        static_cast<float>(this->gui_state.font_load_size), &config) != nullptr) {
                    bool font_api_load_success = this->render_backend.CreateFontsTexture();
                    // Load last added font
                    if (font_api_load_success) {
                        io.FontDefault = io.Fonts->Fonts[(io.Fonts->Fonts.Size - 1)];
                        this->gui_state.default_font_filename = this->gui_state.font_load_name;
                        load_success = true;
                    }
                }
                if (!load_success) {
                    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                        "[GUI] Unable to load font from file '%s' with size %d. [%s, %s, line %d]\n",
                        this->gui_state.font_load_name.c_str(), this->gui_state.font_load_size, __FILE__, __FUNCTION__,
                        __LINE__);
                }
            } else if ((!this->gui_state.font_load_name.empty()) && (this->gui_state.font_load_name != "<unknown>")) {

                std::string imgui_font_string =
                    this->gui_state.font_load_name + ", " + std::to_string(this->gui_state.font_load_size) + "px";
                for (int n = static_cast<int>(this->gui_state.graph_fonts_reserved); n < (io.Fonts->Fonts.Size); n++) {
                    std::string font_name = std::string(io.Fonts->Fonts[n]->GetDebugName());
                    if (font_name == imgui_font_string) {
                        io.FontDefault = io.Fonts->Fonts[n];
                        load_success = true;
                    }
                }
                if (!load_success) {
                    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                        "[GUI] Unable to load font '%s' by name with size %d. The font size might not be available due "
                        "to changed scaling factor. [%s, %s, line %d]\n",
                        this->gui_state.font_load_name.c_str(), this->gui_state.font_load_size, __FILE__, __FUNCTION__,
                        __LINE__);
                }
            }

            this->gui_state.font_load = 0;
        }
    }

    // Process hotkeys --------------------------------------------------------
    if (this->gui_hotkeys[HOTKEY_GUI_SHOW_HIDE_GUI].is_pressed) {
        this->SetVisibility(!this->gui_state.gui_visible);
    }
    if (this->gui_hotkeys[HOTKEY_GUI_EXIT_PROGRAM].is_pressed) {
        this->gui_state.shutdown_triggered = true;
        return true;
    }
    if (this->gui_hotkeys[HOTKEY_GUI_TRIGGER_SCREENSHOT].is_pressed) {
        this->gui_state.screenshot_triggered = true;
    }
    if (this->gui_hotkeys[HOTKEY_GUI_MENU].is_pressed) {
        this->gui_state.menu_visible = !this->gui_state.menu_visible;
    }
    if (this->gui_hotkeys[HOTKEY_GUI_TOGGLE_GRAPH_ENTRY].is_pressed) {
        if (auto graph_ptr = this->win_configurator_ptr->GetGraphCollection().GetRunningGraph()) {
            graph_ptr->ToggleGraphEntry();
        }
    }

    // Reset all hotkeys
    for (auto& hotkey : this->gui_hotkeys) {
        hotkey.second.is_pressed = false;
    }

    // Assume pending changes in scaling as applied  --------------------------
    megamol::gui::gui_scaling.ConsumePendingChange();

    this->win_animation_editor_ptr->RenderAnimation();

    return true;
}


bool GUIManager::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {

    if (!this->gui_state.gui_visible)
        return false;

    ImGui::SetCurrentContext(this->imgui_context);

    ImGuiIO& io = ImGui::GetIO();

    auto imgui_key_index = gui_utils::GlfwKeyToImGuiKey(key);
    io.AddKeyEvent(imgui_key_index, (core::view::KeyAction::PRESS == action));

    io.AddKeyEvent(ImGuiKey_ModCtrl, (mods.equals(megamol::frontend_resources::Modifier::CTRL)));
    io.AddKeyEvent(ImGuiKey_ModShift, (mods.equals(megamol::frontend_resources::Modifier::SHIFT)));
    io.AddKeyEvent(ImGuiKey_ModAlt, (mods.equals(megamol::frontend_resources::Modifier::ALT)));

    // Pass NUM 'Enter' as alternative for 'Return' to ImGui
    if (imgui_key_index == ImGuiKey_KeypadEnter) {
        io.AddKeyEvent(ImGuiKey_Enter, (core::view::KeyAction::PRESS == action));
    }

    // Consume keyboard input if requested by any imgui widget (e.g. text input)
    if (io.WantTextInput) {
        return true;
    }

    return false;
}


bool GUIManager::OnChar(unsigned int codePoint) {

    if (!this->gui_state.gui_visible)
        return false;

    ImGui::SetCurrentContext(this->imgui_context);

    ImGuiIO& io = ImGui::GetIO();
    io.ClearInputCharacters();
    if (codePoint > 0 && codePoint < 0x10000) {
        io.AddInputCharacter((unsigned short)codePoint);
    }

    return false;
}


bool GUIManager::OnMouseMove(double x, double y) {

    if (!this->gui_state.gui_visible)
        return false;

    ImGui::SetCurrentContext(this->imgui_context);

    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(static_cast<float>(x), static_cast<float>(y));

    bool consumed = io.WantCaptureMouse;
    if (!consumed) {
        consumed = this->picking_buffer.ProcessMouseMove(x, y);
    }

    return consumed;
}


bool GUIManager::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {

    if (!this->gui_state.gui_visible)
        return false;

    ImGui::SetCurrentContext(this->imgui_context);

    bool down = (action == core::view::MouseButtonAction::PRESS);
    auto buttonIndex = static_cast<size_t>(button);
    ImGuiIO& io = ImGui::GetIO();

    io.AddKeyEvent(ImGuiKey_ModCtrl, (mods.equals(megamol::frontend_resources::Modifier::CTRL)));
    io.AddKeyEvent(ImGuiKey_ModShift, (mods.equals(megamol::frontend_resources::Modifier::SHIFT)));
    io.AddKeyEvent(ImGuiKey_ModAlt, (mods.equals(megamol::frontend_resources::Modifier::ALT)));

    io.AddMouseButtonEvent(buttonIndex, down);

    bool consumed = io.WantCaptureMouse;
    if (!consumed) {
        consumed = this->picking_buffer.ProcessMouseClick(button, action, mods);
    }

    return consumed;
}


bool GUIManager::OnMouseScroll(double dx, double dy) {

    if (!this->gui_state.gui_visible)
        return false;

    ImGui::SetCurrentContext(this->imgui_context);

    ImGuiIO& io = ImGui::GetIO();
    io.AddMouseWheelEvent(static_cast<float>(dx), static_cast<float>(dy));

    bool consumed = io.WantCaptureMouse;
    return consumed;
}


void megamol::gui::GUIManager::SetScale(float scale) {

    megamol::gui::gui_scaling.Set(scale);
    if (megamol::gui::gui_scaling.PendingChange()) {

        // Reload ImGui style options
        this->gui_state.style_changed = true;

        // Reload and scale all default fonts
        this->gui_state.load_default_fonts = true;

        if (!this->gui_state.default_font_filename.empty()) {
            // Additionally trigger reload of currently used default font that is loaded from file
            /// Need to wait 1 additional frame for scaled font being available!
            this->gui_state.font_load_name = this->gui_state.default_font_filename;
            this->gui_state.font_load_size = static_cast<int>(
                static_cast<float>(this->gui_state.font_load_size) * (megamol::gui::gui_scaling.TransitionFactor()));
            this->gui_state.font_load = 2;
        }
    }
    // Scale all windows
    const auto size_func = [&](AbstractWindow& wc) {
        wc.Config().reset_size *= megamol::gui::gui_scaling.TransitionFactor();
        wc.Config().size *= megamol::gui::gui_scaling.TransitionFactor();
        wc.Config().reset_pos_size = true;
    };
    this->win_collection.EnumWindows(size_func);
}


void megamol::gui::GUIManager::SetClipboardFunc(const char* (*get_clipboard_func)(void* user_data),
    void (*set_clipboard_func)(void* user_data, const char* string), void* user_data) {

    if (this->imgui_context != nullptr) {
        ImGuiIO& io = ImGui::GetIO();
        io.SetClipboardTextFn = set_clipboard_func;
        io.GetClipboardTextFn = get_clipboard_func;
        io.ClipboardUserData = user_data;
    }
}


bool megamol::gui::GUIManager::SynchronizeGraphs(megamol::core::MegaMolGraph& megamol_graph) {

    // Synchronization is not required when no gui element is visible (?)
    if (!this->gui_state.gui_visible)
        return true;

    if (this->win_configurator_ptr->GetGraphCollection().SynchronizeGraphs(megamol_graph)) {

        // Check for new GUI state
        if (!this->gui_state.new_gui_state.empty()) {
            this->state_from_string(this->gui_state.new_gui_state);
            this->gui_state.new_gui_state.clear();
        }

        // Check for new script path name
        if (auto graph_ptr = this->win_configurator_ptr->GetGraphCollection().GetRunningGraph()) {
            std::string script_filename;
            if (!this->gui_state.project_script_paths.empty()) {
                script_filename = this->gui_state.project_script_paths.front();
            }
            if (!script_filename.empty()) {
                auto script_path = std::filesystem::u8path(script_filename);
                graph_ptr->SetFilename(script_path.generic_u8string(), false);
            }
        }

        return true;
    }

    return false;
}


bool GUIManager::create_context() {

    if (this->imgui_context != nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] ImGui context has already been created. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Check for existing context and share FontAtlas with new context (required by ImGui).
    bool other_context_exists = (ImGui::GetCurrentContext() != nullptr);
    ImFontAtlas* font_atlas = nullptr;
    ImFont* default_font = nullptr;
    // Handle multiple ImGui contexts.
    if (other_context_exists) {
        ImGuiIO& current_io = ImGui::GetIO();
        font_atlas = current_io.Fonts;
        default_font = current_io.FontDefault;
        ImGui::GetCurrentContext()->FontAtlasOwnedByContext = false;
    }

    // Create ImGui context ---------------------------------------------------
    IMGUI_CHECKVERSION();
    this->imgui_context = ImGui::CreateContext(font_atlas);
    if (this->imgui_context == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to create ImGui context. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->implot_context = ImPlot::CreateContext();

    // Style settings ---------------------------------------------------------
    ImGui::SetColorEditOptions(ImGuiColorEditFlags_Uint8 | ImGuiColorEditFlags_DisplayRGB |
                               ImGuiColorEditFlags_PickerHueWheel | ImGuiColorEditFlags_AlphaBar |
                               ImGuiColorEditFlags_AlphaPreview);

    // IO settings ------------------------------------------------------------
    ImGuiIO& io = ImGui::GetIO();
    io.IniSavingRate = 5.0f;         // in seconds - unused
    io.IniFilename = nullptr;        // "imgui.ini" - disabled, using own window settings profile
    io.LogFilename = nullptr;        // "imgui_log.txt" - disabled
    io.FontAllowUserScaling = false; // disable font scaling using ctrl + mouse wheel
    /// XXX IO io.ConfigFlags |=  ImGuiConfigFlags_NavEnableKeyboard; // allow keyboard navigation, required for log console -> possible conflict with param hotkeys. io.WantCaptureKeyboard in GUIManager::OnKey blocks all hokeys when GUI window is selected..
    io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors; // GetMouseCursor() is processed in frontend service

/// DOCKING https://github.com/ocornut/imgui/issues/2109
#ifdef IMGUI_HAS_DOCK
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // enable window docking
    io.ConfigDockingWithShift = true;                 // activate docking on pressing 'shift'
#endif

/// MULTI-VIEWPORT https://github.com/ocornut/imgui/issues/1542
#ifdef IMGUI_HAS_VIEWPORT
    /*
    #include "GLFW/glfw3.h"
    #include "imgui_impl_glfw.h"*
    * See ...\build\_deps\imgui-src\examples\example_glfw_opengl3\main.cpp for required setup
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // enable multi-viewport
    // Add ...\plugins\gui\CMakeLists.txt
        #GLFW
        if (USE_GLFW)
            require_external(glfw3) target_link_libraries(${PROJECT_NAME} PRIVATE glfw3) endif()
    // (get glfw_win via WindowManipulation glfw resource)
    ImGui_ImplGlfw_InitForOpenGL(glfw_win, true);
    // ...
    ImGui_ImplGlfw_NewFrame();
    */
#endif

    // Adding additional utf-8 glyph ranges
    // (there is no error if glyph has no representation in font atlas)
    this->gui_state.font_utf8_ranges.clear();
    this->gui_state.font_utf8_ranges.emplace_back(0x0020);
    this->gui_state.font_utf8_ranges.emplace_back(0x03FF); // Basic Latin + Latin Supplement + Greek Alphabet
    this->gui_state.font_utf8_ranges.emplace_back(0x20AC);
    this->gui_state.font_utf8_ranges.emplace_back(0x20AC); // Euro
    this->gui_state.font_utf8_ranges.emplace_back(0x2122);
    this->gui_state.font_utf8_ranges.emplace_back(0x2122); // TM
    this->gui_state.font_utf8_ranges.emplace_back(0x212B);
    this->gui_state.font_utf8_ranges.emplace_back(0x212B); // Angstroem
    this->gui_state.font_utf8_ranges.emplace_back(0x0391);
    this->gui_state.font_utf8_ranges.emplace_back(0); // (range termination)

    // Load initial fonts only once for all imgui contexts --------------------
    if (other_context_exists) {
        // Fonts are already loaded
        if (default_font != nullptr) {
            io.FontDefault = default_font;
        } else {
            // ... else default font is font loaded after configurator fonts -> Index equals number of graph fonts.
            auto default_font_index = static_cast<int>(this->win_configurator_ptr->GetGraphFontScalings().size());
            default_font_index = std::min(default_font_index, io.Fonts->Fonts.Size - 1);
            io.FontDefault = io.Fonts->Fonts[default_font_index];
        }
    } else {
        this->gui_state.load_default_fonts = true;
    }

    return true;
}


bool GUIManager::destroy_context() {

    if (this->render_backend.IsBackendInitialized()) {
        if (this->imgui_context != nullptr) {

            // Handle multiple ImGui contexts.
            if (megamol::gui::gui_context_count < 2) {
                ImGui::SetCurrentContext(this->imgui_context);
                // Shutdown API only if only one context is left
                this->render_backend.ShutdownBackend();
                // Last context should delete font atlas
                ImGui::GetCurrentContext()->FontAtlasOwnedByContext = true;
            }

            if (this->implot_context != nullptr) {
                ImPlot::DestroyContext(this->implot_context);
            }
            ImGui::DestroyContext(this->imgui_context);
            megamol::gui::gui_context_count--;
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Destroyed ImGui context.");
        }
        this->imgui_context = nullptr;
    }

    // Reset global state
    this->init_state();

    return true;
}


void megamol::gui::GUIManager::load_default_fonts() {

    if (this->render_backend.SupportsCustomFonts()) {

        ImGuiIO& io = ImGui::GetIO();
        io.Fonts->Clear();

        const auto graph_font_scalings = this->win_configurator_ptr->GetGraphFontScalings();
        this->gui_state.graph_fonts_reserved = graph_font_scalings.size();

        const float default_font_size = (12.0f * megamol::gui::gui_scaling.Get());
        ImFontConfig config;
        config.OversampleH = 4;
        config.OversampleV = 4;
        config.GlyphRanges = this->gui_state.font_utf8_ranges.data();

        // Get other known fonts
        std::vector<std::string> font_paths;
        std::string configurator_font_path;
        std::string default_font_path;
        auto get_preset_font_path = [&](const std::string& directory) {
            std::string font_path =
                megamol::core::utility::FileUtils::SearchFileRecursive(directory, GUI_FILENAME_FONT_DEFAULT_ROBOTOSANS);
            if (!font_path.empty()) {
                font_paths.emplace_back(font_path);
                configurator_font_path = font_path;
                default_font_path = font_path;
            }
            font_path = megamol::core::utility::FileUtils::SearchFileRecursive(
                directory, GUI_FILENAME_FONT_DEFAULT_SOURCECODEPRO);
            if (!font_path.empty()) {
                font_paths.emplace_back(font_path);
            }
        };
        for (auto& resource_directory : megamol::gui::gui_resource_paths) {
            get_preset_font_path(resource_directory);
        }

        // Configurator Graph Font: Add default font at first n indices for exclusive use in configurator graph.
        /// Workaround: Using different font sizes for different graph zooming factors to improve font readability when
        /// zooming.
        if (configurator_font_path.empty()) {
            for (unsigned int i = 0; i < this->gui_state.graph_fonts_reserved; i++) {
                io.Fonts->AddFontDefault(&config);
            }
        } else {
            for (unsigned int i = 0; i < this->gui_state.graph_fonts_reserved; i++) {
                io.Fonts->AddFontFromFileTTF(
                    configurator_font_path.c_str(), default_font_size * graph_font_scalings[i], &config);
            }
        }

        // Add other fonts for gui.
        io.Fonts->AddFontDefault(&config);
        io.FontDefault = io.Fonts->Fonts[(io.Fonts->Fonts.Size - 1)];
        for (auto& font_path : font_paths) {
            io.Fonts->AddFontFromFileTTF(font_path.c_str(), default_font_size, &config);
            if (default_font_path == font_path) {
                io.FontDefault = io.Fonts->Fonts[(io.Fonts->Fonts.Size - 1)];
            }
        }

        this->gui_state.default_font_filename.clear();
    }

    this->render_backend.CreateFontsTexture();
}


void GUIManager::draw_menu() {

    if (!this->gui_state.menu_visible)
        return;

    ImGuiStyle& style = ImGui::GetStyle();

    ImGui::BeginMainMenuBar();

    // FILE -------------------------------------------------------------------
    if (ImGui::BeginMenu("File")) {

        if (ImGui::MenuItem("Load Project", this->gui_hotkeys[HOTKEY_GUI_LOAD_PROJECT].keycode.ToString().c_str())) {
            this->gui_state.open_popup_load = true;
        }
        this->tooltip.ToolTip("Project will be added to currently running project.");
        if (ImGui::MenuItem("Save Project", this->gui_hotkeys[HOTKEY_GUI_SAVE_PROJECT].keycode.ToString().c_str())) {
            this->gui_state.open_popup_save = true;
        }
        if (ImGui::MenuItem("Exit", this->gui_hotkeys[HOTKEY_GUI_EXIT_PROGRAM].keycode.ToString().c_str())) {
            this->gui_state.shutdown_triggered = true;
        }
        ImGui::EndMenu();
    }
    ImGui::Separator();

    // WINDOWS ----------------------------------------------------------------
    if (ImGui::BeginMenu("Windows")) {
        ImGui::MenuItem(
            "Menu", this->gui_hotkeys[HOTKEY_GUI_MENU].keycode.ToString().c_str(), &this->gui_state.menu_visible);
        const auto func = [&](AbstractWindow& wc) {
            bool registered_window = (wc.Config().hotkey.key != core::view::Key::KEY_UNKNOWN);
            if (registered_window) {
                ImGui::MenuItem(wc.Name().c_str(), wc.Config().hotkey.ToString().c_str(), &wc.Config().show);
            } else {
                // Custom unregistered parameter window
                if (ImGui::BeginMenu(wc.Name().c_str())) {
                    std::string GUI_MENU_label = "Show";
                    if (wc.Config().show)
                        GUI_MENU_label = "Hide";
                    if (ImGui::MenuItem(GUI_MENU_label.c_str(), wc.Config().hotkey.ToString().c_str(), nullptr)) {
                        wc.Config().show = !wc.Config().show;
                    }
                    // Enable option to delete custom newly created parameter windows
                    if (wc.WindowID() == AbstractWindow::WINDOW_ID_PARAMETERS) {
                        if (ImGui::MenuItem("Delete Window")) {
                            this->gui_state.win_delete_hash_id = wc.Hash();
                        }
                    }
                    ImGui::EndMenu();
                }
            }
        };
        this->win_collection.EnumWindows(func);

        ImGui::Separator();

        if (ImGui::MenuItem(
                "Show/Hide All Windows", this->gui_hotkeys[HOTKEY_GUI_SHOW_HIDE_GUI].keycode.ToString().c_str())) {
            this->gui_state.gui_hide_next_frame = 3;
        }

/// DOCKING
#ifdef IMGUI_HAS_DOCK
        if (ImGui::MenuItem("Windows Docking Preset")) {
            this->gui_state.load_docking_preset = true;
        }
#endif

        ImGui::Separator();

        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Transparency");
        this->tooltip.Marker("Alpha value of window background");
        ImGui::SameLine();
        if (ImGui::SliderFloat("###window_transparency", &this->gui_state.window_alpha, 0.0f, 1.0f, "%.2f")) {
            ImGuiStyle& style = ImGui::GetStyle();
            style.Colors[ImGuiCol_WindowBg].w = this->gui_state.window_alpha;
            style.Colors[ImGuiCol_ChildBg].w = this->gui_state.window_alpha;
            ;
        }

        ImGui::EndMenu();
    }
    ImGui::Separator();

    // SCREENSHOT -------------------------------------------------------------
    if (ImGui::BeginMenu("Screenshot")) {
        if (ImGui::MenuItem("Select File Name", this->gui_state.screenshot_filepath.c_str())) {
            this->gui_state.open_popup_screenshot = true;
        }
        if (ImGui::MenuItem("Trigger", this->gui_hotkeys[HOTKEY_GUI_TRIGGER_SCREENSHOT].keycode.ToString().c_str())) {
            this->gui_state.screenshot_triggered = true;
        }
        ImGui::EndMenu();
    }
    ImGui::Separator();

    // RENDER -----------------------------------------------------------------
    if (ImGui::BeginMenu("Projects")) {
        for (auto& graph_ptr : this->win_configurator_ptr->GetGraphCollection().GetGraphs()) {

            const std::string label_id = "##" + std::to_string(graph_ptr->UID());
            if (ImGui::BeginMenu((graph_ptr->Name() + label_id).c_str())) {

                bool running = graph_ptr->IsRunning();
                std::string button_label = "graph_running_button" + label_id;
                if (megamol::gui::ButtonWidgets::OptionButton(ButtonWidgets::ButtonStyle::POINT_CIRCLE, button_label,
                        ((running) ? ("Running") : ("Run")), running, running)) {
                    if (!running) {
                        this->win_configurator_ptr->GetGraphCollection().RequestNewRunningGraph(graph_ptr->UID());
                    }
                }
                ImGui::Separator();

                ImGui::TextDisabled("Graph Entries:");
                for (auto& module_ptr : graph_ptr->Modules()) {
                    if (module_ptr->IsView()) {
                        if (ImGui::MenuItem(module_ptr->FullName().c_str(), "", module_ptr->IsGraphEntry())) {
                            if (!module_ptr->IsGraphEntry()) {
                                graph_ptr->AddGraphEntry(module_ptr, graph_ptr->GenerateUniqueGraphEntryName());
                            } else {
                                graph_ptr->RemoveGraphEntry(module_ptr);
                            }
                        }
                    }
                }
                if (ImGui::MenuItem("Toggle Graph Entry",
                        this->gui_hotkeys[HOTKEY_GUI_TOGGLE_GRAPH_ENTRY].keycode.ToString().c_str())) {
                    this->gui_hotkeys[HOTKEY_GUI_TOGGLE_GRAPH_ENTRY].is_pressed = true;
                }

                ImGui::EndMenu();
            }
            ImGui::AlignTextToFramePadding();
        }
        ImGui::EndMenu();
    }
    ImGui::Separator();

    // SETTINGS ---------------------------------------------------------------
    if (ImGui::BeginMenu("Settings")) {

        if (ImGui::BeginMenu("Style")) {

            if (ImGui::MenuItem(
                    "ImGui Dark Colors", nullptr, (this->gui_state.style == GUIManager::Styles::DarkColors))) {
                this->gui_state.style = GUIManager::Styles::DarkColors;
                this->gui_state.style_changed = true;
            }
            if (ImGui::MenuItem(
                    "ImGui LightColors", nullptr, (this->gui_state.style == GUIManager::Styles::LightColors))) {
                this->gui_state.style = GUIManager::Styles::LightColors;
                this->gui_state.style_changed = true;
            }
            if (ImGui::MenuItem(
                    "Corporate Gray", nullptr, (this->gui_state.style == GUIManager::Styles::CorporateGray))) {
                this->gui_state.style = GUIManager::Styles::CorporateGray;
                this->gui_state.style_changed = true;
            }
            if (ImGui::MenuItem(
                    "Corporate White", nullptr, (this->gui_state.style == GUIManager::Styles::CorporateWhite))) {
                this->gui_state.style = GUIManager::Styles::CorporateWhite;
                this->gui_state.style_changed = true;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Scale")) {
            ImGui::TextUnformatted("GUI Scaling Factor:");
            ImGui::PushItemWidth(ImGui::GetFrameHeight() * 5.0f);
            ImGui::InputFloat("###scale_input", &this->gui_state.scale_input_float_buffer, 0.1f, 0.5f, "%.2f",
                ImGuiInputTextFlags_None);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                this->SetScale(this->gui_state.scale_input_float_buffer);
            } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                this->gui_state.scale_input_float_buffer = megamol::gui::gui_scaling.Get();
            }
            ImGui::PopItemWidth();
            ImGui::Separator();
            ImGui::TextUnformatted("Presets:");
            if (ImGui::MenuItem("100%", nullptr, (this->gui_state.scale_input_float_buffer == 1.0f))) {
                this->SetScale(1.0f);
            }
            if (ImGui::MenuItem("150%", nullptr, (this->gui_state.scale_input_float_buffer == 1.5f))) {
                this->SetScale(1.5f);
            }
            if (ImGui::MenuItem("200%", nullptr, (this->gui_state.scale_input_float_buffer == 2.0f))) {
                this->SetScale(2.0f);
            }
            if (ImGui::MenuItem("250%", nullptr, (this->gui_state.scale_input_float_buffer == 2.5f))) {
                this->SetScale(2.5f);
            }
            if (ImGui::MenuItem("300%", nullptr, (this->gui_state.scale_input_float_buffer == 3.0f))) {
                this->SetScale(3.0f);
            }

            ImGui::EndMenu();
        }

        if (ImGui::MenuItem("Font")) {
            this->gui_state.open_popup_font = true;
        }
#ifdef MEGAMOL_USE_PROFILING
        if (ImGui::MenuItem(this->perf_logging->active ? "Pause performance logging" : "Resume performance logging")) {
            this->perf_logging->active = !this->perf_logging->active;
        }
#endif

        ImGui::EndMenu();
    }
    ImGui::Separator();

    // HELP -------------------------------------------------------------------
    if (ImGui::BeginMenu("Help")) {
        if (ImGui::BeginMenu("Debug")) {
            ImGui::MenuItem("Dear ImGui Metrics", nullptr, &this->gui_state.show_imgui_metrics);
            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("About")) {
            this->gui_state.open_popup_about = true;
        }
        ImGui::EndMenu();
    }
    ImGui::Separator();

    ImGui::EndMainMenuBar();
}


void megamol::gui::GUIManager::draw_popups() {

    ImGuiStyle& style = ImGui::GetStyle();
    ImGuiIO& io = ImGui::GetIO();

    // Externally registered pop-ups
    auto popup_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar;
    for (auto it = this->popup_collection.begin(); it != this->popup_collection.end(); it++) {
        if (it->second.open_flag.expired() || (it->second.open_flag.lock() == nullptr)) {
            this->popup_collection.erase(it);
            break;
        }
        if (*(it->second.open_flag.lock())) {
            *(it->second.open_flag.lock()) = false;
            ImGui::OpenPopup(it->first.c_str());
        }
        if (ImGui::BeginPopupModal(it->first.c_str(), nullptr, popup_flags)) {
            it->second.draw_callback();
            if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }

    // Externally registered notifications
    for (auto it = this->notification_collection.begin(); it != this->notification_collection.end(); it++) {
        if (it->second.open_flag.expired() || (it->second.open_flag.lock() == nullptr)) {
            this->notification_collection.erase(it);
            break;
        }
        if (*it->second.open_flag.lock()) {
            *it->second.open_flag.lock() = false;
            if (!it->second.disable) {
                ImGui::OpenPopup(it->first.c_str());
            }
        }
        if (ImGui::BeginPopupModal(it->first.c_str(), nullptr, popup_flags)) {
            ImGui::TextUnformatted(it->second.message.c_str());
            bool close = false;
            if (ImGui::Button("Ok")) {
                close = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Ok - Disable further notifications.")) {
                close = true;
                // Disable further notifications
                it->second.disable = true;
            }
            if (close || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }


    // FONT
    if (this->gui_state.open_popup_font) {
        this->gui_state.open_popup_font = false;
        ImGui::OpenPopup("Font");
    }
    bool open = true;
    if (ImGui::BeginPopupModal("Font", &open, (ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove))) {

        bool close_popup = false;

        ImFont* font_current = ImGui::GetFont();
        ImGui::TextUnformatted("Select available font:");
        if (ImGui::BeginCombo("###select_available_font", font_current->GetDebugName())) {
            /// first fonts until index this->graph_fonts_reserved are exclusively used by graph in configurator
            for (int n = static_cast<int>(this->gui_state.graph_fonts_reserved); n < io.Fonts->Fonts.Size; n++) {
                if (ImGui::Selectable(io.Fonts->Fonts[n]->GetDebugName(), (io.Fonts->Fonts[n] == font_current))) {
                    io.FontDefault = io.Fonts->Fonts[n];
                    this->gui_state.default_font_filename.clear();
                    /// UX: close_popup = true;
                }
            }
            ImGui::EndCombo();
        }

        ImGui::Separator();

        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Load new font from TTF file:");

        std::string label("Font Size");
        ImGui::InputInt(label.c_str(), &this->gui_state.font_load_size, 1, 10, ImGuiInputTextFlags_None);
        // Validate font size
        if (this->gui_state.font_load_size <= 5) {
            this->gui_state.font_load_size = 5; // minimum valid font size
        }

        ImGui::BeginGroup();

        float widget_width = ImGui::CalcItemWidth() - (ImGui::GetFrameHeightWithSpacing() + style.ItemSpacing.x);
        ImGui::PushItemWidth(widget_width);

        this->file_browser.Button_Select(this->gui_state.font_input_string_buffer, {"ttf"},
            megamol::core::param::FilePathParam::Flag_File_RestrictExtension);
        ImGui::SameLine();
        ImGui::InputText("Font Filename (.ttf)", &this->gui_state.font_input_string_buffer, ImGuiInputTextFlags_None);
        ImGui::PopItemWidth();
        // Validate font file before offering load button
        bool valid_file = megamol::core::utility::FileUtils::FileWithExtensionExists<std::string>(
            this->gui_state.font_input_string_buffer, std::string("ttf"));

        if (!valid_file) {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Add Font")) {
            this->gui_state.font_load_name = this->gui_state.font_input_string_buffer;
            this->gui_state.font_load = 1;
            /// UX: close_popup = true;
        }
        std::string help("Same font can be loaded multiple times with different font size.");
        this->tooltip.Marker(help);
        if (!valid_file) {
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextColored(GUI_COLOR_TEXT_ERROR, "Please enter valid font file name.");
        }

        ImGui::Separator();
        if (ImGui::Button("Close") || close_popup || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndGroup();

        ImGui::EndPopup();
    }

    // ABOUT
    if (this->gui_state.open_popup_about) {
        this->gui_state.open_popup_about = false;
        ImGui::OpenPopup("About");
    }
    open = true;
    if (ImGui::BeginPopupModal("About", &open, (ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove))) {

        const std::string email("megamol@visus.uni-stuttgart.de");
        const std::string web_link("https://megamol.org/");
        const std::string github_link("https://github.com/UniStuttgart-VISUS/megamol");
        const std::string docu_link("https://github.com/UniStuttgart-VISUS/megamol/tree/master/plugins/gui");
        const std::string imgui_link("https://github.com/ocornut/imgui");

        const std::string mmstr = std::string("MegaMol - Version ") +
                                  megamol::core::utility::buildinfo::MEGAMOL_VERSION() + ("\ngit# ") +
                                  megamol::core::utility::buildinfo::MEGAMOL_GIT_HASH() + ("\n");
        const std::string mailstr = std::string("Contact: ") + email;
        const std::string webstr = std::string("Web: ") + web_link;
        const std::string gitstr = std::string("Git-Hub: ") + github_link;
        const std::string imguistr = ("Dear ImGui - Version ") + std::string(IMGUI_VERSION) + ("\n");
        const std::string imguigitstr = std::string("Git-Hub: ") + imgui_link;
        std::string commit_date = megamol::core::utility::buildinfo::MEGAMOL_GIT_LAST_COMMIT_DATE();
        std::string year;
        if (commit_date.empty()) {
            // Backup when commit date is no available via build script info
            std::time_t t = std::time(0);
            std::tm* now = std::localtime(&t);
            year = std::to_string(now->tm_year + 1900);
        } else {
            year = commit_date.substr(0, 4);
        }
        const std::string about =
            "Copyright (C) 2009-" + year + " by University of Stuttgart (VISUS).\nAll rights reserved.";

        ImGui::TextUnformatted(mmstr.c_str());

        if (ImGui::Button("Copy E-Mail")) {
            ImGui::SetClipboardText(email.c_str());
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(mailstr.c_str());

        if (ImGui::Button("Copy Website")) {
            ImGui::SetClipboardText(web_link.c_str());
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(webstr.c_str());

        if (ImGui::Button("Copy GitHub###megamol_copy_github")) {
            ImGui::SetClipboardText(github_link.c_str());
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(gitstr.c_str());

        ImGui::Separator();
        ImGui::TextUnformatted(imguistr.c_str());
        if (ImGui::Button("Copy GitHub###imgui_copy_github")) {
            ImGui::SetClipboardText(imgui_link.c_str());
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(imguigitstr.c_str());

        ImGui::Separator();
        ImGui::TextUnformatted(about.c_str());

        ImGui::Separator();
        if (ImGui::Button("Close") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    // Save project pop-up
    if (auto graph_ptr = this->win_configurator_ptr->GetGraphCollection().GetRunningGraph()) {
        this->gui_state.open_popup_save |= this->gui_hotkeys[HOTKEY_GUI_SAVE_PROJECT].is_pressed;
        this->gui_state.open_popup_save |= this->win_configurator_ptr->ConsumeTriggeredGlobalProjectSave();

        auto filename = graph_ptr->GetFilename();
        bool popup_failed = false;
        // Default for saving gui state and parameter values
        bool save_all_param_values = true;
        bool save_gui_state = false;
        if (this->file_browser.PopUp_Save("Save Running Project", filename, this->gui_state.open_popup_save, {"lua"},
                megamol::core::param::FilePathParam::Flag_File_ToBeCreatedWithRestrExts, save_gui_state,
                save_all_param_values)) {
            std::string state_str;
            if (save_gui_state) {
                state_str = this->project_to_lua_string(true);
            }
            popup_failed = !this->win_configurator_ptr->GetGraphCollection().SaveProjectToFile(
                graph_ptr->UID(), filename, state_str, save_all_param_values);
        }
        PopUps::Minimal(
            "Failed to Save Project", popup_failed, "See console log output for more information.", "Cancel");
    }
    this->gui_hotkeys[HOTKEY_GUI_SAVE_PROJECT].is_pressed = false;

    // Load project pop-up
    std::string filename;
    this->gui_state.open_popup_load |= this->gui_hotkeys[HOTKEY_GUI_LOAD_PROJECT].is_pressed;
    if (this->file_browser.PopUp_Load("Load Project", filename, this->gui_state.open_popup_load, {"lua", "png"},
            megamol::core::param::FilePathParam::Flag_File_RestrictExtension)) {
        // Redirect project loading request to Lua_Wrapper_service and load new project to megamol graph
        /// GUI graph and GUI state are updated at next synchronization
        this->gui_state.request_load_projet_file = filename;
    }
    this->gui_hotkeys[HOTKEY_GUI_LOAD_PROJECT].is_pressed = false;

    // File name for screenshot pop-up
    auto dummy = false;
    if (this->file_browser.PopUp_Save("File Name for Screenshot", this->gui_state.screenshot_filepath,
            this->gui_state.open_popup_screenshot, {"png"},
            megamol::core::param::FilePathParam::Flag_File_ToBeCreatedWithRestrExts, dummy, dummy)) {
        this->gui_state.screenshot_filepath_id = 0;
    }

    // ImGui metrics
    if (this->gui_state.show_imgui_metrics) {
        ImGui::ShowMetricsWindow(&this->gui_state.show_imgui_metrics);
    }
}


void megamol::gui::GUIManager::load_preset_window_docking(ImGuiID global_docking_id) {

/// DOCKING
#ifdef IMGUI_HAS_DOCK

    // Create preset using DockBuilder
    /// https://github.com/ocornut/imgui/issues/2109#issuecomment-426204357
    //   -------------------------------
    //   |      |                      |
    //   | prop |       main           |
    //   |      |                      |
    //   |      |                      |
    //   |______|______________________|
    //   |           bottom            |
    //   -------------------------------

    ImGuiIO& io = ImGui::GetIO();
    auto dockspace_size = io.DisplaySize;
    ImGui::DockBuilderRemoveNode(global_docking_id);                            // Clear out existing layout
    ImGui::DockBuilderAddNode(global_docking_id, ImGuiDockNodeFlags_DockSpace); // Add empty node
    ImGui::DockBuilderSetNodeSize(global_docking_id, dockspace_size);
    // Define new dock spaces
    ImGuiID dock_id_main = global_docking_id; // This variable will track the document node, however we are not using it
                                              // here as we aren't docking anything into it.
    ImGuiID dock_id_bottom = ImGui::DockBuilderSplitNode(dock_id_main, ImGuiDir_Down, 0.25f, nullptr, &dock_id_main);
    ImGuiID dock_id_prop = ImGui::DockBuilderSplitNode(dock_id_main, ImGuiDir_Left, 0.25f, nullptr, &dock_id_main);

    const auto func = [&](AbstractWindow& wc) {
        switch (wc.WindowID()) {
        case (AbstractWindow::WINDOW_ID_MAIN_PARAMETERS): {
            ImGui::DockBuilderDockWindow(wc.FullWindowTitle().c_str(), dock_id_prop);
        } break;
        // case (AbstractWindow::WINDOW_ID_TRANSFER_FUNCTION): {
        //    ImGui::DockBuilderDockWindow(wc.FullWindowTitle().c_str(), dock_id_prop);
        //} break;
        case (AbstractWindow::WINDOW_ID_CONFIGURATOR): {
            ImGui::DockBuilderDockWindow(wc.FullWindowTitle().c_str(), dock_id_main);
        } break;
        case (AbstractWindow::WINDOW_ID_LOGCONSOLE): {
            ImGui::DockBuilderDockWindow(wc.FullWindowTitle().c_str(), dock_id_bottom);
        } break;
        default:
            break;
        }
    };
    this->win_collection.EnumWindows(func);

    ImGui::DockBuilderFinish(global_docking_id);
#endif
}


void megamol::gui::GUIManager::load_imgui_settings_from_string(const std::string& imgui_settings) {

/// DOCKING
#ifdef IMGUI_HAS_DOCK
    this->gui_state.load_docking_preset = true;
    if (!imgui_settings.empty()) {
        ImGui::LoadIniSettingsFromMemory(imgui_settings.c_str(), imgui_settings.size());
        this->gui_state.load_docking_preset = false;
    }
#endif
}


std::string megamol::gui::GUIManager::save_imgui_settings_to_string() const {

/// DOCKING
#ifdef IMGUI_HAS_DOCK
    size_t buffer_size = 0;
    const char* buffer = ImGui::SaveIniSettingsToMemory(&buffer_size);
    if (buffer == nullptr) {
        return std::string();
    }
    std::string imgui_settings(buffer, buffer_size);
    return imgui_settings;
#else
    return std::string();
#endif
}


std::string megamol::gui::GUIManager::project_to_lua_string(bool as_lua) {

    std::string state_str;
    if (this->state_to_string(state_str)) {
        std::string return_state_str;

        if (as_lua) {
            return_state_str += std::string(GUI_START_TAG_SET_GUI_VISIBILITY) +
                                ((this->gui_state.gui_visible) ? ("true") : ("false")) +
                                std::string(GUI_END_TAG_SET_GUI_VISIBILITY) + "\n";

            return_state_str += std::string(GUI_START_TAG_SET_GUI_SCALE) +
                                std::to_string(megamol::gui::gui_scaling.Get()) +
                                std::string(GUI_END_TAG_SET_GUI_SCALE) + "\n";

            return_state_str +=
                std::string(GUI_START_TAG_SET_GUI_STATE) + state_str + std::string(GUI_END_TAG_SET_GUI_STATE) + "\n";
        } else {
            return_state_str += state_str;
        }

        return return_state_str;
    }
    return std::string();
}


bool megamol::gui::GUIManager::state_from_string(const std::string& state) {

    try {
        nlohmann::json state_json = nlohmann::json::parse(state);
        if (!state_json.is_object()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Invalid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        // Read GUI state
        for (auto& header_item : state_json.items()) {
            if (header_item.key() == GUI_JSON_TAG_GUI) {
                auto state_str = header_item.value();
                megamol::core::utility::get_json_value<bool>(
                    state_str, {"menu_visible"}, &this->gui_state.menu_visible);
                int style = 0;
                megamol::core::utility::get_json_value<int>(state_str, {"style"}, &style);
                this->gui_state.style = static_cast<GUIManager::Styles>(style);
                this->gui_state.style_changed = true;

                megamol::core::utility::get_json_value<std::string>(
                    state_str, {"font_file_name"}, &this->gui_state.font_load_name);
                megamol::core::utility::get_json_value<int>(state_str, {"font_size"}, &this->gui_state.font_load_size);
                this->gui_state.font_load = 2;

                std::string imgui_settings;
                megamol::core::utility::get_json_value<std::string>(state_str, {"imgui_settings"}, &imgui_settings);
                this->load_imgui_settings_from_string(imgui_settings);
                megamol::core::utility::get_json_value<float>(
                    state_str, {"global_win_background_alpha"}, &this->gui_state.window_alpha);
            }
        }

        // Read window configurations
        this->win_collection.StateFromJSON(state_json);

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Read GUI state from JSON.");
#endif // GUI_VERBOSE
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON Error - Unable to read state from JSON. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


bool megamol::gui::GUIManager::state_to_string(std::string& out_state) {

    ImGuiIO& io = ImGui::GetIO();

    try {
        out_state.clear();
        nlohmann::json json_state;

        json_state[GUI_JSON_TAG_GUI]["menu_visible"] = this->gui_state.menu_visible;
        json_state[GUI_JSON_TAG_GUI]["style"] = static_cast<int>(this->gui_state.style);
        json_state[GUI_JSON_TAG_GUI]["font_file_name"] =
            ((this->gui_state.default_font_filename.empty()) ? (this->extract_fontname(io.FontDefault->GetDebugName()))
                                                             : (this->gui_state.default_font_filename));
        json_state[GUI_JSON_TAG_GUI]["font_size"] = ImGui::GetFontSize();
        json_state[GUI_JSON_TAG_GUI]["imgui_settings"] = this->save_imgui_settings_to_string();
        json_state[GUI_JSON_TAG_GUI]["global_win_background_alpha"] = this->gui_state.window_alpha;

        // Write window configurations
        this->win_collection.StateToJSON(json_state);

        out_state = json_state.dump();

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Wrote GUI state to JSON.");
#endif // GUI_VERBOSE

    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON Error - Unable to write state to JSON. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


bool megamol::gui::GUIManager::create_unique_screenshot_filename(std::string& inout_filepath) {

    // Check for existing file
    bool created_filepath = false;
    if (!inout_filepath.empty()) {
        auto ret_filepath = inout_filepath;
        do {
            // Create new filename with iterating suffix
            std::string filename = megamol::core::utility::FileUtils::GetFilePathStem<std::string>(ret_filepath);
            std::string id_separator = "_";
            bool new_separator = false;
            auto separator_index = filename.find_last_of(id_separator);
            if (separator_index != std::string::npos) {
                auto last_id_str = filename.substr(separator_index + 1);
                std::istringstream(last_id_str) >> this->gui_state.screenshot_filepath_id; // 0 if failed
                if (this->gui_state.screenshot_filepath_id == 0) {
                    new_separator = true;
                }
                this->gui_state.screenshot_filepath_id++;
                if (new_separator) {
                    ret_filepath =
                        filename + id_separator + std::to_string(this->gui_state.screenshot_filepath_id) + ".png";
                } else {
                    ret_filepath = filename.substr(0, separator_index + 1) +
                                   std::to_string(this->gui_state.screenshot_filepath_id) + ".png";
                }
            } else {
                ret_filepath =
                    filename + id_separator + std::to_string(this->gui_state.screenshot_filepath_id) + ".png";
            }
        } while (megamol::core::utility::FileUtils::FileExists<std::string>(ret_filepath));
        inout_filepath = std::filesystem::u8path(ret_filepath).generic_u8string();
        created_filepath = true;
    }
    return created_filepath;
}


void GUIManager::RegisterWindow(
    const std::string& window_name, std::function<void(AbstractWindow::BasicConfig&)> const& callback) {

    this->win_collection.AddWindow(window_name, callback);
}


void GUIManager::RegisterPopUp(
    const std::string& name, std::weak_ptr<bool> open, const std::function<void()>& callback) {

    this->popup_collection[name] = PopUpData{open, const_cast<std::function<void()>&>(callback)};
}


void GUIManager::RegisterNotification(const std::string& name, std::weak_ptr<bool> open, const std::string& message) {

    this->notification_collection[name] = NotificationData{open, false, message};
}


std::string GUIManager::extract_fontname(const std::string& imgui_fontname) const {

    auto return_fontname = std::string(imgui_fontname);
    auto sep_index = return_fontname.find(',');
    return_fontname = return_fontname.substr(0, sep_index);
    return return_fontname;
}


void GUIManager::RegisterHotkeys(
    megamol::core::view::CommandRegistry& cmdregistry, megamol::core::MegaMolGraph& megamolgraph) {

    if (auto win_hkeditor_ptr = this->win_collection.GetWindow<HotkeyEditor>()) {
        win_hkeditor_ptr->RegisterHotkeys(&cmdregistry, &megamolgraph, &this->win_collection, &this->gui_hotkeys);
    }
}


void megamol::gui::GUIManager::setRequestedResources(
    std::shared_ptr<frontend_resources::FrontendResourcesMap> const& resources) {
    win_collection.setRequestedResources(resources);
}
