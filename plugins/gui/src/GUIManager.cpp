/*
 * GUIManager.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "GUIManager.h"
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"
#include "mmcore/versioninfo.h"
#include "widgets/ButtonWidgets.h"
#include "widgets/CorporateGreyStyle.h"
#include "widgets/CorporateWhiteStyle.h"
#include "widgets/DefaultStyle.h"
#include "windows/PerformanceMonitor.h"


using namespace megamol;
using namespace megamol::gui;


GUIManager::GUIManager()
        : hotkeys()
        , context(nullptr)
        , initialized_api(megamol::gui::GUIImGuiAPI::NONE)
        , gui_state()
        , win_collection()
        , popup_collection()
        , notification_collection()
        , win_configurator_ptr(nullptr)
        , file_browser()
        , tooltip()
        , picking_buffer() {

    // Init hotkeys
    this->hotkeys[HOTKEY_GUI_TRIGGER_SCREENSHOT] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F2, core::view::Modifier::NONE), false};
    this->hotkeys[HOTKEY_GUI_TOGGLE_GRAPH_ENTRY] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F3, core::view::Modifier::NONE), false};
    this->hotkeys[HOTKEY_GUI_EXIT_PROGRAM] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F4, core::view::Modifier::ALT), false};
    this->hotkeys[HOTKEY_GUI_MENU] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F12, core::view::Modifier::NONE), false};
    this->hotkeys[HOTKEY_GUI_SAVE_PROJECT] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_S, core::view::Modifier::CTRL), false};
    this->hotkeys[HOTKEY_GUI_LOAD_PROJECT] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_L, core::view::Modifier::CTRL), false};
    this->hotkeys[HOTKEY_GUI_SHOW_HIDE_GUI] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_G, core::view::Modifier::CTRL), false};

    this->win_configurator_ptr = this->win_collection.GetWindow<Configurator>();
    if (this->win_configurator_ptr == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "[GUI] BUG - Failed to get shortcut pointer to configurator.");
    }

    this->init_state();
}


GUIManager::~GUIManager() {

    this->destroy_context();
}


void megamol::gui::GUIManager::init_state() {

    this->gui_state.gui_visible = true;
    this->gui_state.gui_visible_post = true;
    this->gui_state.gui_restore_hidden_windows.clear();
    this->gui_state.gui_hide_next_frame = 0;
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
    this->gui_state.menu_visible = true;
    this->gui_state.graph_fonts_reserved = 0;
    this->gui_state.shutdown_triggered = false;
    this->gui_state.screenshot_triggered = false;
    this->gui_state.screenshot_filepath = "megamol_screenshot.png";
    this->create_unique_screenshot_filename(this->gui_state.screenshot_filepath);
    this->gui_state.screenshot_filepath_id = 0;
    this->gui_state.font_load = 0;
    this->gui_state.font_load_filename = "";
    this->gui_state.font_load_size = 12;
    this->gui_state.request_load_projet_file = "";
    this->gui_state.stat_averaged_fps = 0.0f;
    this->gui_state.stat_averaged_ms = 0.0f;
    this->gui_state.stat_frame_count = 0;
    this->gui_state.load_docking_preset = true;
}


bool GUIManager::CreateContext(GUIImGuiAPI imgui_api) {

    // Check prerequisities for requested API
    switch (imgui_api) {
    case (GUIImGuiAPI::OPEN_GL): {
        bool prerequisities_given = true;
#ifdef _WIN32 // Windows
        HDC ogl_current_display = ::wglGetCurrentDC();
        HGLRC ogl_current_context = ::wglGetCurrentContext();
        if (ogl_current_display == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(
                megamol::core::utility::log::Log::LEVEL_ERROR, "[GUI] There is no OpenGL rendering context available.");
            prerequisities_given = false;
        }
        if (ogl_current_context == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
                "[GUI] There is no current OpenGL rendering context available from the calling thread.");
            prerequisities_given = false;
        }
#else
        /// XXX The following throws segfault if OpenGL is not loaded yet:
        // Display* gl_current_display = ::glXGetCurrentDisplay();
        // GLXContext ogl_current_context = ::glXGetCurrentContext();
        /// XXX Is there a better way to check existing OpenGL context?
        if (glXGetCurrentDisplay == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(
                megamol::core::utility::log::Log::LEVEL_ERROR, "[GUI] There is no OpenGL rendering context available.");
            prerequisities_given = false;
        }
        if (glXGetCurrentContext == nullptr) {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
                "[GUI] There is no current OpenGL rendering context available from the calling thread.");
            prerequisities_given = false;
        }
#endif // _WIN32
        if (!prerequisities_given) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Failed to create ImGui context for OpenGL API. [%s, %s, line %d]\n<<< HINT: Check if "
                "project contains view module. >>>",
                __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
    } break;
    default: {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] ImGui API is not supported. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    } break;
    }

    // Create ImGui Context
    bool other_context_exists = (ImGui::GetCurrentContext() != nullptr);
    if (this->create_context()) {

        // Initialize ImGui API
        if (!other_context_exists) {
            switch (imgui_api) {
            case (GUIImGuiAPI::OPEN_GL): {
                // Init OpenGL for ImGui
                if (ImGui_ImplOpenGL3_Init(nullptr)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Created ImGui context for Open GL.");
                } else {
                    this->destroy_context();
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Unable to initialize OpenGL for ImGui. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                        __LINE__);
                    return false;
                }

            } break;
            default: {
                this->destroy_context();
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] ImGui API is not supported. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                return false;
            } break;
            }
        }

        this->initialized_api = imgui_api;
        megamol::gui::gui_context_count++;
        ImGui::SetCurrentContext(this->context);
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
        this->gui_state.gui_visible = false;
    }

    // Check for initialized imgui api
    if (this->initialized_api == GUIImGuiAPI::NONE) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No ImGui API initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Check for existing imgui context
    if (this->context == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No valid ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Set ImGui context
    ImGui::SetCurrentContext(this->context);

    // Process hotkeys
    if (this->hotkeys[HOTKEY_GUI_SHOW_HIDE_GUI].is_pressed) {
        if (this->gui_state.gui_visible) {
            this->gui_state.gui_hide_next_frame = 2;
        } else {
            // Show GUI after it was hidden (before early exit!)
            // Restore window 'open' state (Always restore at least HOTKEY_GUI_MENU)
            this->gui_state.menu_visible = true;
            const auto func = [&](AbstractWindow& wc) {
                if (std::find(this->gui_state.gui_restore_hidden_windows.begin(),
                        this->gui_state.gui_restore_hidden_windows.end(),
                        wc.Name()) != this->gui_state.gui_restore_hidden_windows.end()) {
                    wc.Config().show = true;
                }
            };
            this->win_collection.EnumWindows(func);
            this->gui_state.gui_restore_hidden_windows.clear();
            this->gui_state.gui_visible = true;
        }
    }
    if (this->hotkeys[HOTKEY_GUI_EXIT_PROGRAM].is_pressed) {
        this->gui_state.shutdown_triggered = true;
        return true;
    }
    if (this->hotkeys[HOTKEY_GUI_TRIGGER_SCREENSHOT].is_pressed) {
        this->gui_state.screenshot_triggered = true;
    }
    if (this->hotkeys[HOTKEY_GUI_MENU].is_pressed) {
        this->gui_state.menu_visible = !this->gui_state.menu_visible;
    }
    if (this->hotkeys[HOTKEY_GUI_TOGGLE_GRAPH_ENTRY].is_pressed) {
        if (auto graph_ptr = this->win_configurator_ptr->GetGraphCollection().GetRunningGraph()) {
            graph_ptr->ToggleGraphEntry();
        }
    }

    // Delayed font loading for resource directories being available via resource in frontend
    if (this->gui_state.load_default_fonts) {
        this->load_default_fonts();
        this->gui_state.load_default_fonts = false;
    }

    // Required to prevent change in gui drawing between pre and post draw
    this->gui_state.gui_visible_post = this->gui_state.gui_visible;
    // Early exit when pre step should be omitted
    if (!this->gui_state.gui_visible) {
        return true;
    }

    // Set stuff for next frame --------------------------------------------
    ImGuiIO& io = ImGui::GetIO();

    io.DisplaySize = ImVec2(window_size.x, window_size.y);
    if ((window_size.x > 0.0f) && (window_size.y > 0.0f)) {
        io.DisplayFramebufferScale = ImVec2(framebuffer_size.x / window_size.x, framebuffer_size.y / window_size.y);
    }

    if ((instance_time - this->gui_state.last_instance_time) < 0.0) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Current instance time results in negative time delta. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
    }
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

    // Update windows
    this->win_collection.Update();

    // Delete window
    if (this->gui_state.win_delete_hash_id != 0) {
        this->win_collection.DeleteWindow(this->gui_state.win_delete_hash_id);
        this->gui_state.win_delete_hash_id = 0;
    }

    // Start new ImGui frame --------------------------------------------------
    ImGui_ImplOpenGL3_NewFrame();
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

    // Early exit when post step should be omitted
    if (!this->gui_state.gui_visible_post) {
        return true;
    }
    // Check for initialized imgui api
    if (this->initialized_api == GUIImGuiAPI::NONE) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No ImGui API initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Check for existing imgui context
    if (this->context == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] No valid ImGui context available. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Set ImGui context
    ImGui::SetCurrentContext(this->context);
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    ////////// DRAW GUI ///////////////////////////////////////////////////////
    try {

        // Main HOTKEY_GUI_MENU ---------------------------------------------------------------
        this->draw_menu();

        // Draw Windows ------------------------------------------------------------
        this->win_collection.Draw(this->gui_state.menu_visible);

        // Draw Pop-ups ------------------------------------------------------------
        this->draw_popups();

        // Draw global parameter widgets -------------------------------------------
        if (auto graph_ptr = this->win_configurator_ptr->GetGraphCollection().GetRunningGraph()) {
            /// ! Only enabled in second frame if interaction objects are added during first frame !
            this->picking_buffer.EnableInteraction(glm::vec2(io.DisplaySize.x, io.DisplaySize.y));
            graph_ptr->DrawGlobalParameterWidgets(
                this->picking_buffer, this->win_collection.GetWindow<TransferFunctionEditor>());
            this->picking_buffer.DisableInteraction();
        }

    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error... [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }

    ///////////////////////////////////////////////////////////////////////////

    /// TODO - SEPARATE RENDERING OF OPENGL-STUFF DEPENDING ON AVAILABLE API?!

    // Render the current ImGui frame ------------------------------------------
    glViewport(0, 0, static_cast<GLsizei>(io.DisplaySize.x), static_cast<GLsizei>(io.DisplaySize.y));
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Reset all hotkeys ------------------------------------------------------
    for (auto& hotkey : this->hotkeys) {
        hotkey.second.is_pressed = false;
    }

    // Assume pending changes in scaling as applied  --------------------------
    megamol::gui::gui_scaling.ConsumePendingChange();

    // Hide GUI if it is currently shown --------------------------------------
    if (this->gui_state.gui_visible) {
        if (this->gui_state.gui_hide_next_frame == 2) {
            // First frame
            this->gui_state.gui_hide_next_frame--;
            // Save 'open' state of windows for later restore. Closing all windows before omitting GUI rendering is
            // required to set right ImGui state for mouse handling
            this->gui_state.gui_restore_hidden_windows.clear();
            const auto func = [&](AbstractWindow& wc) {
                if (wc.Config().show) {
                    this->gui_state.gui_restore_hidden_windows.push_back(wc.Name());
                    wc.Config().show = false;
                }
            };
            this->win_collection.EnumWindows(func);
        } else if (this->gui_state.gui_hide_next_frame == 1) {
            // Second frame
            this->gui_state.gui_hide_next_frame = 0;
            this->gui_state.gui_visible = false;
        }
    }

    // Loading new font -------------------------------------------------------
    // (after first imgui frame for default fonts being available)
    if (this->gui_state.font_load > 1) {
        this->gui_state.font_load--;
    } else if (this->gui_state.font_load == 1) {
        bool load_success = false;
        if (megamol::core::utility::FileUtils::FileWithExtensionExists<std::string>(
                this->gui_state.font_load_filename, std::string("ttf"))) {
            ImFontConfig config;
            config.OversampleH = 4;
            config.OversampleV = 4;
            config.GlyphRanges = this->gui_state.font_utf8_ranges.data();
            gui_utils::Utf8Encode(this->gui_state.font_load_filename);
            if (io.Fonts->AddFontFromFileTTF(this->gui_state.font_load_filename.c_str(),
                    static_cast<float>(this->gui_state.font_load_size), &config) != nullptr) {
                bool font_api_load_success = false;
                switch (this->initialized_api) {
                case (GUIImGuiAPI::OPEN_GL): {
                    font_api_load_success = ImGui_ImplOpenGL3_CreateFontsTexture();
                } break;
                default: {
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] ImGui API is not supported. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                } break;
                }
                // Load last added font
                if (font_api_load_success) {
                    io.FontDefault = io.Fonts->Fonts[(io.Fonts->Fonts.Size - 1)];
                    load_success = true;
                }
            }
            gui_utils::Utf8Decode(this->gui_state.font_load_filename);
        } else if ((!this->gui_state.font_load_filename.empty()) &&
                   (this->gui_state.font_load_filename != "<unknown>")) {
            std::string imgui_font_string =
                this->gui_state.font_load_filename + ", " + std::to_string(this->gui_state.font_load_size) + "px";
            for (int n = static_cast<int>(this->gui_state.graph_fonts_reserved); n < (io.Fonts->Fonts.Size); n++) {
                std::string font_name = std::string(io.Fonts->Fonts[n]->GetDebugName());
                gui_utils::Utf8Decode(font_name);
                if (font_name == imgui_font_string) {
                    io.FontDefault = io.Fonts->Fonts[n];
                    load_success = true;
                }
            }
        }
        // if (!load_success) {
        //    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
        //        "[GUI] Unable to load font '%s' with size %d (NB: ImGui default font ProggyClean.ttf can only be "
        //        "loaded with predefined size 13). [%s, %s, line %d]\n",
        //        this->gui_state.font_load_filename.c_str(), this->gui_state.font_load_size, __FILE__, __FUNCTION__,
        //        __LINE__);
        //}
        this->gui_state.font_load = 0;
    }

    return true;
}


bool GUIManager::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {

    ImGui::SetCurrentContext(this->context);

    ImGuiIO& io = ImGui::GetIO();

    bool last_return_key = io.KeysDown[static_cast<size_t>(core::view::Key::KEY_ENTER)];
    bool last_num_enter_key = io.KeysDown[static_cast<size_t>(core::view::Key::KEY_KP_ENTER)];

    auto keyIndex = static_cast<size_t>(key);
    switch (action) {
    case core::view::KeyAction::PRESS:
        io.KeysDown[keyIndex] = true;
        break;
    case core::view::KeyAction::RELEASE:
        io.KeysDown[keyIndex] = false;
        break;
    default:
        break;
    }
    io.KeyCtrl = mods.test(core::view::Modifier::CTRL);
    io.KeyShift = mods.test(core::view::Modifier::SHIFT);
    io.KeyAlt = mods.test(core::view::Modifier::ALT);

    // Pass NUM 'Enter' as alternative for 'Return' to ImGui
    bool cur_return_key = ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_ENTER));
    bool cur_num_enter_key = ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_KP_ENTER));
    bool return_pressed = (!last_return_key && cur_return_key);
    bool enter_pressed = (!last_num_enter_key && cur_num_enter_key);
    io.KeysDown[static_cast<size_t>(core::view::Key::KEY_ENTER)] = (return_pressed || enter_pressed);

    bool hotkeyPressed = false;

    // Check for additional text modification hotkeys
    if (action == core::view::KeyAction::RELEASE) {
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_A)] = false;
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_C)] = false;
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_V)] = false;
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_X)] = false;
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_Y)] = false;
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_Z)] = false;
    }
    hotkeyPressed = true;
    if (io.KeyCtrl && ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_A))) {
        keyIndex = static_cast<size_t>(GuiTextModHotkeys::CTRL_A);
    } else if (io.KeyCtrl && ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_C))) {
        keyIndex = static_cast<size_t>(GuiTextModHotkeys::CTRL_C);
    } else if (io.KeyCtrl && ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_V))) {
        keyIndex = static_cast<size_t>(GuiTextModHotkeys::CTRL_V);
    } else if (io.KeyCtrl && ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_X))) {
        keyIndex = static_cast<size_t>(GuiTextModHotkeys::CTRL_X);
    } else if (io.KeyCtrl && ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_Y))) {
        keyIndex = static_cast<size_t>(GuiTextModHotkeys::CTRL_Y);
    } else if (io.KeyCtrl && ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_Z))) {
        keyIndex = static_cast<size_t>(GuiTextModHotkeys::CTRL_Z);
    } else {
        hotkeyPressed = false;
    }
    if (hotkeyPressed && (action == core::view::KeyAction::PRESS)) {
        io.KeysDown[keyIndex] = true;
        return true;
    }

    // GUI
    for (auto& hotkey : this->hotkeys) {
        if (this->is_hotkey_pressed(hotkey.second.keycode)) {
            hotkey.second.is_pressed = true;
            hotkeyPressed = true;
        }
    }
    // Hotkeys of window(s)
    const auto windows_func = [&](AbstractWindow& wc) {
        // Check Window Hotkey
        bool windowHotkeyPressed = this->is_hotkey_pressed(wc.Config().hotkey);
        if (windowHotkeyPressed) {
            wc.Config().show = !wc.Config().show;
        }
        hotkeyPressed |= windowHotkeyPressed;

        // Check for additional window hotkeys
        for (auto& hotkey : wc.GetHotkeys()) {
            if (this->is_hotkey_pressed(hotkey.second.keycode)) {
                hotkey.second.is_pressed = true;
                hotkeyPressed = true;
            }
        }
    };
    this->win_collection.EnumWindows(windows_func);

    if (hotkeyPressed)
        return true;

    // Always consume keyboard input if requested by any imgui widget (e.g. text input).
    // User expects hotkey priority of text input thus needs to be processed before parameter hotkeys.
    if (io.WantTextInput) { /// io.WantCaptureKeyboard
        return true;
    }

    // Check for parameter hotkeys
    hotkeyPressed = false;
    if (auto graph_ptr = this->win_configurator_ptr->GetGraphCollection().GetRunningGraph()) {
        for (auto& module_ptr : graph_ptr->Modules()) {
            // Break loop after first occurrence of parameter hotkey
            if (hotkeyPressed) {
                break;
            }
            for (auto& param : module_ptr->Parameters()) {
                if (param.Type() == ParamType_t::BUTTON) {
                    auto keyCode = param.GetStorage<megamol::core::view::KeyCode>();
                    if (this->is_hotkey_pressed(keyCode)) {
                        // Sync directly button action to parameter in core
                        /// Does not require syncing of graphs
                        if (param.CoreParamPtr() != nullptr) {
                            param.CoreParamPtr()->setDirty();
                        }
                        /// param.ForceSetValueDirty();
                        hotkeyPressed = true;
                    }
                }
            }
        }
    }

    return hotkeyPressed;
}


bool GUIManager::OnChar(unsigned int codePoint) {
    ImGui::SetCurrentContext(this->context);

    ImGuiIO& io = ImGui::GetIO();
    io.ClearInputCharacters();
    if (codePoint > 0 && codePoint < 0x10000) {
        io.AddInputCharacter((unsigned short) codePoint);
    }

    return false;
}


bool GUIManager::OnMouseMove(double x, double y) {

    ImGui::SetCurrentContext(this->context);

    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2(static_cast<float>(x), static_cast<float>(y));

    auto hoverFlags = ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenDisabled |
                      ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem;

    // Always consumed if any imgui windows is hovered.
    bool consumed = ImGui::IsWindowHovered(hoverFlags);
    if (!consumed) {
        consumed = this->picking_buffer.ProcessMouseMove(x, y);
    }

    return consumed;
}


bool GUIManager::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {

    ImGui::SetCurrentContext(this->context);

    bool down = (action == core::view::MouseButtonAction::PRESS);
    auto buttonIndex = static_cast<size_t>(button);
    ImGuiIO& io = ImGui::GetIO();

    auto hoverFlags = ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenDisabled |
                      ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem;

    io.MouseDown[buttonIndex] = down;

    // Always consumed if any imgui windows is hovered.
    bool consumed = ImGui::IsWindowHovered(hoverFlags);
    if (!consumed) {
        consumed = this->picking_buffer.ProcessMouseClick(button, action, mods);
    }

    return consumed;
}


bool GUIManager::OnMouseScroll(double dx, double dy) {

    ImGui::SetCurrentContext(this->context);

    ImGuiIO& io = ImGui::GetIO();
    io.MouseWheelH += (float) dx;
    io.MouseWheel += (float) dy;

    auto hoverFlags = ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenDisabled |
                      ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem;

    // Always consumed if any imgui windows is hovered.
    bool consumed = ImGui::IsWindowHovered(hoverFlags);
    return consumed;
}


void megamol::gui::GUIManager::SetScale(float scale) {

    megamol::gui::gui_scaling.Set(scale);
    if (megamol::gui::gui_scaling.PendingChange()) {

        // Reload ImGui style options
        this->gui_state.style_changed = true;
        // Reload and scale all fonts
        this->gui_state.load_default_fonts = true;

        // Additionally trigger reload of currently used default font
        /// Need to wait 1 additional frame for scaled font being available!
        this->gui_state.font_load = 2;
        this->gui_state.font_load_size = static_cast<int>(
            static_cast<float>(this->gui_state.font_load_size) * (megamol::gui::gui_scaling.TransitionFactor()));
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

    if (this->context != nullptr) {
        ImGuiIO& io = ImGui::GetIO();
        io.SetClipboardTextFn = set_clipboard_func;
        io.GetClipboardTextFn = get_clipboard_func;
        io.ClipboardUserData = user_data;
    }
}


bool megamol::gui::GUIManager::SynchronizeRunningGraph(
    megamol::core::MegaMolGraph& megamol_graph, megamol::core::CoreInstance& core_instance) {

    // Synchronization is not required when no gui element is visible
    if (!this->gui_state.gui_visible)
        return true;

    // 1) Load all known calls from core instance ONCE ---------------------------
    if (!this->win_configurator_ptr->GetGraphCollection().LoadCallStock(core_instance)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Failed to load call stock once. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Load all known modules from core instance ONCE
    if (!this->win_configurator_ptr->GetGraphCollection().LoadModuleStock(core_instance)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Failed to load module stock once. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    bool synced = false;
    bool sync_success = false;
    GraphPtr_t graph_ptr = this->win_configurator_ptr->GetGraphCollection().GetRunningGraph();
    // 2a) Either synchronize GUI Graph -> Core Graph ... ---------------------
    if (!synced && (graph_ptr != nullptr)) {
        bool graph_sync_success = true;

        Graph::QueueAction action;
        Graph::QueueData data;
        while (graph_ptr->PopSyncQueue(action, data)) {
            synced = true;
            switch (action) {
            case (Graph::QueueAction::ADD_MODULE): {
                graph_sync_success &= megamol_graph.CreateModule(data.class_name, data.name_id);
            } break;
            case (Graph::QueueAction::RENAME_MODULE): {
                bool rename_success = megamol_graph.RenameModule(data.name_id, data.rename_id);
                graph_sync_success &= rename_success;
            } break;
            case (Graph::QueueAction::DELETE_MODULE): {
                graph_sync_success &= megamol_graph.DeleteModule(data.name_id);
            } break;
            case (Graph::QueueAction::ADD_CALL): {
                graph_sync_success &= megamol_graph.CreateCall(data.class_name, data.caller, data.callee);
            } break;
            case (Graph::QueueAction::DELETE_CALL): {
                graph_sync_success &= megamol_graph.DeleteCall(data.caller, data.callee);
            } break;
            case (Graph::QueueAction::CREATE_GRAPH_ENTRY): {
                megamol_graph.SetGraphEntryPoint(data.name_id);
            } break;
            case (Graph::QueueAction::REMOVE_GRAPH_ENTRY): {
                megamol_graph.RemoveGraphEntryPoint(data.name_id);
            } break;
            default:
                break;
            }
        }
        if (!graph_sync_success) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Failed to synchronize gui graph with core graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
        }
        sync_success &= graph_sync_success;
    }

    // 2b) ... OR (exclusive or) synchronize Core Graph -> GUI Graph ----------
    if (!synced) {
        // Creates new graph at first call
        ImGuiID running_graph_uid = (graph_ptr != nullptr) ? (graph_ptr->UID()) : (GUI_INVALID_ID);
        bool graph_sync_success = this->win_configurator_ptr->GetGraphCollection().LoadUpdateProjectFromCore(
            running_graph_uid, megamol_graph);
        if (!graph_sync_success) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Failed to synchronize core graph with gui graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
        }

        // Check for new GUI state
        if (!this->gui_state.new_gui_state.empty()) {
            this->state_from_string(this->gui_state.new_gui_state);
            this->gui_state.new_gui_state.clear();
        }

        // Check for new script path name
        if (graph_sync_success) {
            if (auto synced_graph_ptr = this->win_configurator_ptr->GetGraphCollection().GetRunningGraph()) {
                std::string script_filename;
                // Get project filename from lua state of frontend service
                if (!this->gui_state.project_script_paths.empty()) {
                    script_filename = this->gui_state.project_script_paths.front();
                }
                // Load GUI state from project file when project file changed
                if (!script_filename.empty()) {
                    synced_graph_ptr->SetFilename(script_filename, false);
                }
            }
        }

        sync_success &= graph_sync_success;
    }

    // 3) Synchronize parameter values -------------------------------------------
    if (graph_ptr != nullptr) {
        bool param_sync_success = true;
        for (auto& module_ptr : graph_ptr->Modules()) {
            for (auto& param : module_ptr->Parameters()) {

                // Try to connect gui parameters to newly created parameters of core modules
                if (param.CoreParamPtr().IsNull()) {
                    auto module_name = module_ptr->FullName();
                    megamol::core::Module* core_module_ptr = nullptr;
                    core_module_ptr = megamol_graph.FindModule(module_name).get();
                    // Connect pointer of new parameters of core module to parameters in gui module
                    if (core_module_ptr != nullptr) {
                        auto se = core_module_ptr->ChildList_End();
                        for (auto si = core_module_ptr->ChildList_Begin(); si != se; ++si) {
                            auto param_slot = dynamic_cast<megamol::core::param::ParamSlot*>((*si).get());
                            if (param_slot != nullptr) {
                                std::string param_full_name(param_slot->FullName().PeekBuffer());
                                for (auto& parameter : module_ptr->Parameters()) {
                                    if (gui_utils::CaseInsensitiveStringCompare(
                                            parameter.FullNameCore(), param_full_name)) {
                                        megamol::gui::Parameter::ReadNewCoreParameterToExistingParameter(
                                            (*param_slot), parameter, true, false, true);
                                    }
                                }
                            }
                        }
                    }
#ifdef GUI_VERBOSE
                    if (param.CoreParamPtr().IsNull()) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Unable to connect core parameter to gui parameter. [%s, %s, line %d]\n", __FILE__,
                            __FUNCTION__, __LINE__);
                    }
#endif // GUI_VERBOSE
                }

                if (!param.CoreParamPtr().IsNull()) {
                    // Write changed gui state to core parameter
                    if (param.IsGUIStateDirty()) {
                        param_sync_success &=
                            megamol::gui::Parameter::WriteCoreParameterGUIState(param, param.CoreParamPtr());
                        param.ResetGUIStateDirty();
                    }
                    // Write changed parameter value to core parameter
                    if (param.IsValueDirty()) {
                        param_sync_success &=
                            megamol::gui::Parameter::WriteCoreParameterValue(param, param.CoreParamPtr());
                        param.ResetValueDirty();
                    }
                    // Read current parameter value and GUI state fro core parameter
                    param_sync_success &= megamol::gui::Parameter::ReadCoreParameterToParameter(
                        param.CoreParamPtr(), param, false, false);
                }
            }
        }
#ifdef GUI_VERBOSE
        if (!param_sync_success) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] Failed to synchronize parameter values. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
#endif // GUI_VERBOSE
        sync_success &= param_sync_success;
    }
    return sync_success;
}


bool GUIManager::create_context() {

    if (this->initialized_api != GUIImGuiAPI::NONE) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] ImGui context has alreday been created. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return true;
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
    this->context = ImGui::CreateContext(font_atlas);
    if (this->context == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to create ImGui context. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Style settings ---------------------------------------------------------
    ImGui::SetColorEditOptions(ImGuiColorEditFlags_Uint8 | ImGuiColorEditFlags_DisplayRGB |
                               ImGuiColorEditFlags_PickerHueWheel | ImGuiColorEditFlags_AlphaBar |
                               ImGuiColorEditFlags_AlphaPreview);

    // IO settings ------------------------------------------------------------
    ImGuiIO& io = ImGui::GetIO();
    io.IniSavingRate = 5.0f;                              //  in seconds - unused
    io.IniFilename = nullptr;                             // "imgui.ini" - disabled, using own window settings profile
    io.LogFilename = nullptr;                             // "imgui_log.txt" - disabled
    io.FontAllowUserScaling = false;                      // disable font scaling using ctrl + mouse wheel
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // allow keyboard navigation
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

    // ImGui Key Map
    io.KeyMap[ImGuiKey_Tab] = static_cast<int>(core::view::Key::KEY_TAB);
    io.KeyMap[ImGuiKey_LeftArrow] = static_cast<int>(core::view::Key::KEY_LEFT);
    io.KeyMap[ImGuiKey_RightArrow] = static_cast<int>(core::view::Key::KEY_RIGHT);
    io.KeyMap[ImGuiKey_UpArrow] = static_cast<int>(core::view::Key::KEY_UP);
    io.KeyMap[ImGuiKey_DownArrow] = static_cast<int>(core::view::Key::KEY_DOWN);
    io.KeyMap[ImGuiKey_PageUp] = static_cast<int>(core::view::Key::KEY_PAGE_UP);
    io.KeyMap[ImGuiKey_PageDown] = static_cast<int>(core::view::Key::KEY_PAGE_DOWN);
    io.KeyMap[ImGuiKey_Home] = static_cast<int>(core::view::Key::KEY_HOME);
    io.KeyMap[ImGuiKey_End] = static_cast<int>(core::view::Key::KEY_END);
    io.KeyMap[ImGuiKey_Insert] = static_cast<int>(core::view::Key::KEY_INSERT);
    io.KeyMap[ImGuiKey_Delete] = static_cast<int>(core::view::Key::KEY_DELETE);
    io.KeyMap[ImGuiKey_Backspace] = static_cast<int>(core::view::Key::KEY_BACKSPACE);
    io.KeyMap[ImGuiKey_Space] = static_cast<int>(core::view::Key::KEY_SPACE);
    io.KeyMap[ImGuiKey_Enter] = static_cast<int>(core::view::Key::KEY_ENTER);
    io.KeyMap[ImGuiKey_Escape] = static_cast<int>(core::view::Key::KEY_ESCAPE);
    io.KeyMap[ImGuiKey_A] = static_cast<int>(GuiTextModHotkeys::CTRL_A);
    io.KeyMap[ImGuiKey_C] = static_cast<int>(GuiTextModHotkeys::CTRL_C);
    io.KeyMap[ImGuiKey_V] = static_cast<int>(GuiTextModHotkeys::CTRL_V);
    io.KeyMap[ImGuiKey_X] = static_cast<int>(GuiTextModHotkeys::CTRL_X);
    io.KeyMap[ImGuiKey_Y] = static_cast<int>(GuiTextModHotkeys::CTRL_Y);
    io.KeyMap[ImGuiKey_Z] = static_cast<int>(GuiTextModHotkeys::CTRL_Z);

    // Init global state -------------------------------------------------------
    this->init_state();

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

    if (this->initialized_api != GUIImGuiAPI::NONE) {
        if (this->context != nullptr) {

            // Handle multiple ImGui contexts.
            if (megamol::gui::gui_context_count < 2) {
                ImGui::SetCurrentContext(this->context);
                // Shutdown API only if only one context is left
                switch (this->initialized_api) {
                case (GUIImGuiAPI::OPEN_GL):
                    ImGui_ImplOpenGL3_Shutdown();
                    break;
                default:
                    break;
                }
                // Last context should delete font atlas
                ImGui::GetCurrentContext()->FontAtlasOwnedByContext = true;
            }

            ImGui::DestroyContext(this->context);
            megamol::gui::gui_context_count--;
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Destroyed ImGui context.");
        }
        this->context = nullptr;
        this->initialized_api = GUIImGuiAPI::NONE;
    }

    return true;
}


void megamol::gui::GUIManager::load_default_fonts() {

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
            megamol::core::utility::FileUtils::SearchFileRecursive(directory, GUI_DEFAULT_FONT_ROBOTOSANS);
        if (!font_path.empty()) {
            font_paths.emplace_back(font_path);
            configurator_font_path = font_path;
            default_font_path = font_path;
        }
        font_path = megamol::core::utility::FileUtils::SearchFileRecursive(directory, GUI_DEFAULT_FONT_SOURCECODEPRO);
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

    // Set default if there is no pending font load request otherwise
    if (this->gui_state.font_load == 0) {
        this->gui_state.font_load_filename = default_font_path;
        this->gui_state.font_load_size = static_cast<int>(default_font_size);
    }

    switch (this->initialized_api) {
    case (GUIImGuiAPI::NONE): {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Fonts can only be loaded after API was initialized. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
    } break;
    case (GUIImGuiAPI::OPEN_GL): {
        ImGui_ImplOpenGL3_CreateFontsTexture();
    } break;
    default: {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] ImGui API is not supported. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    } break;
    }
}


void GUIManager::draw_menu() {

    if (!this->gui_state.menu_visible)
        return;

    ImGuiStyle& style = ImGui::GetStyle();

    ImGui::BeginMainMenuBar();

    // FILE -------------------------------------------------------------------
    if (ImGui::BeginMenu("File")) {

        if (ImGui::MenuItem("Load Project", this->hotkeys[HOTKEY_GUI_LOAD_PROJECT].keycode.ToString().c_str())) {
            this->gui_state.open_popup_load = true;
        }
        this->tooltip.ToolTip("Project will be added to currently running project.");
        if (ImGui::MenuItem("Save Project", this->hotkeys[HOTKEY_GUI_SAVE_PROJECT].keycode.ToString().c_str())) {
            this->gui_state.open_popup_save = true;
        }
        if (ImGui::MenuItem("Exit", this->hotkeys[HOTKEY_GUI_EXIT_PROGRAM].keycode.ToString().c_str())) {
            this->gui_state.shutdown_triggered = true;
        }
        ImGui::EndMenu();
    }
    ImGui::Separator();

    // WINDOWS ----------------------------------------------------------------
    if (ImGui::BeginMenu("Windows")) {
        ImGui::MenuItem(
            "Menu", this->hotkeys[HOTKEY_GUI_MENU].keycode.ToString().c_str(), &this->gui_state.menu_visible);
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
                "Show/Hide All Windows", this->hotkeys[HOTKEY_GUI_SHOW_HIDE_GUI].keycode.ToString().c_str())) {
            this->gui_state.gui_hide_next_frame = 2;
        }

/// DOCKING
#ifdef IMGUI_HAS_DOCK
        if (ImGui::MenuItem("Windows Docking Preset")) {
            this->gui_state.load_docking_preset = true;
        }
#endif
        ImGui::EndMenu();
    }
    ImGui::Separator();

    // SCREENSHOT -------------------------------------------------------------
    if (ImGui::BeginMenu("Screenshot")) {
        if (ImGui::MenuItem("Select Filename", this->gui_state.screenshot_filepath.c_str())) {
            this->gui_state.open_popup_screenshot = true;
        }
        if (ImGui::MenuItem("Trigger", this->hotkeys[HOTKEY_GUI_TRIGGER_SCREENSHOT].keycode.ToString().c_str())) {
            this->gui_state.screenshot_triggered = true;
        }
        ImGui::EndMenu();
    }
    ImGui::Separator();

    // RENDER -----------------------------------------------------------------
    if (ImGui::BeginMenu("Projects")) {
        for (auto& graph_ptr : this->win_configurator_ptr->GetGraphCollection().GetGraphs()) {

            if (ImGui::BeginMenu(graph_ptr->Name().c_str())) {

                bool running = graph_ptr->IsRunning();
                std::string button_label = "graph_running_button" + std::to_string(graph_ptr->UID());
                if (megamol::gui::ButtonWidgets::OptionButton(
                        button_label, ((running) ? ("Running") : ("Run")), running, running)) {
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
                                // Remove all graph entries
                                for (auto& rem_module_ptr : graph_ptr->Modules()) {
                                    if (rem_module_ptr->IsView() && rem_module_ptr->IsGraphEntry()) {
                                        rem_module_ptr->SetGraphEntryName("");
                                        Graph::QueueData queue_data;
                                        queue_data.name_id = rem_module_ptr->FullName();
                                        graph_ptr->PushSyncQueue(Graph::QueueAction::REMOVE_GRAPH_ENTRY, queue_data);
                                    }
                                }
                                // Add new graph entry
                                module_ptr->SetGraphEntryName(graph_ptr->GenerateUniqueGraphEntryName());
                                Graph::QueueData queue_data;
                                queue_data.name_id = module_ptr->FullName();
                                graph_ptr->PushSyncQueue(Graph::QueueAction::CREATE_GRAPH_ENTRY, queue_data);
                            } else {
                                module_ptr->SetGraphEntryName("");
                                Graph::QueueData queue_data;
                                queue_data.name_id = module_ptr->FullName();
                                graph_ptr->PushSyncQueue(Graph::QueueAction::REMOVE_GRAPH_ENTRY, queue_data);
                            }
                        }
                    }
                }
                if (ImGui::MenuItem("Toggle Graph Entry",
                        this->hotkeys[HOTKEY_GUI_TOGGLE_GRAPH_ENTRY].keycode.ToString().c_str())) {
                    this->hotkeys[HOTKEY_GUI_TOGGLE_GRAPH_ENTRY].is_pressed = true;
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

        if (ImGui::BeginMenu("Font")) {

            ImGuiIO& io = ImGui::GetIO();
            ImFont* font_current = ImGui::GetFont();
            if (ImGui::BeginCombo("Select Available Font", font_current->GetDebugName())) {
                /// first fonts until index this->graph_fonts_reserved are exclusively used by graph in configurator
                for (int n = static_cast<int>(this->gui_state.graph_fonts_reserved); n < io.Fonts->Fonts.Size; n++) {
                    if (ImGui::Selectable(io.Fonts->Fonts[n]->GetDebugName(), (io.Fonts->Fonts[n] == font_current))) {
                        io.FontDefault = io.Fonts->Fonts[n];
                        this->gui_state.font_load_filename = this->extract_fontname(io.FontDefault->GetDebugName());
                        this->gui_state.font_load_size = static_cast<int>(io.FontDefault->FontSize);
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::Separator();

            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted("Load Font from File");
            std::string help("Same font can be loaded multiple times with different font size.");
            this->tooltip.Marker(help);

            std::string label("Font Size");
            ImGui::InputInt(label.c_str(), &this->gui_state.font_load_size, 1, 10, ImGuiInputTextFlags_None);
            // Validate font size
            if (this->gui_state.font_load_size <= 5) {
                this->gui_state.font_load_size = 5; // minimum valid font size
            }

            ImGui::BeginGroup();
            float widget_width = ImGui::CalcItemWidth() - (ImGui::GetFrameHeightWithSpacing() + style.ItemSpacing.x);
            ImGui::PushItemWidth(widget_width);
            this->file_browser.Button_Select({"ttf"}, this->gui_state.font_load_filename, true);
            ImGui::SameLine();
            gui_utils::Utf8Encode(this->gui_state.font_load_filename);
            ImGui::InputText("Font Filename (.ttf)", &this->gui_state.font_load_filename, ImGuiInputTextFlags_None);
            gui_utils::Utf8Decode(this->gui_state.font_load_filename);
            ImGui::PopItemWidth();
            // Validate font file before offering load button
            bool valid_file = megamol::core::utility::FileUtils::FileWithExtensionExists<std::string>(
                this->gui_state.font_load_filename, std::string("ttf"));
            if (!valid_file) {
                megamol::gui::gui_utils::PushReadOnly();
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            }
            if (ImGui::Button("Add Font")) {
                this->gui_state.font_load = 1;
            }
            if (!valid_file) {
                ImGui::PopItemFlag();
                megamol::gui::gui_utils::PopReadOnly();
                ImGui::SameLine();
                ImGui::TextColored(GUI_COLOR_TEXT_ERROR, "Please enter valid font file name.");
            }
            ImGui::EndGroup();

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Scale")) {
            float scale = megamol::gui::gui_scaling.Get();
            if (ImGui::RadioButton("100%", (scale == 1.0f))) {
                this->SetScale(1.0f);
            }
            if (ImGui::RadioButton("150%", (scale == 1.5f))) {
                this->SetScale(1.5f);
            }
            if (ImGui::RadioButton("200%", (scale == 2.0f))) {
                this->SetScale(2.0f);
            }
            if (ImGui::RadioButton("250%", (scale == 2.5f))) {
                this->SetScale(2.5f);
            }
            if (ImGui::RadioButton("300%", (scale == 3.0f))) {
                this->SetScale(3.0f);
            }

            ImGui::EndMenu();
        }

        ImGui::EndMenu();
    }
    ImGui::Separator();

    // HELP -------------------------------------------------------------------
    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("About")) {
            this->gui_state.open_popup_about = true;
        }
        ImGui::EndMenu();
    }
    ImGui::Separator();

    ImGui::EndMainMenuBar();
}


void megamol::gui::GUIManager::draw_popups() {

    // Draw pop-ups defined in windows
    this->win_collection.PopUps();

    // Externally registered pop-ups
    auto popup_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar;
    for (auto& popup_map : this->popup_collection) {
        if ((*popup_map.second.first)) {
            ImGui::OpenPopup(popup_map.first.c_str());
            (*popup_map.second.first) = false;
        }
        if (ImGui::BeginPopupModal(popup_map.first.c_str(), nullptr, popup_flags)) {
            popup_map.second.second();
            if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape))) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }

    // Externally registered notifications
    for (auto& popup_map : this->notification_collection) {
        if (!std::get<1>(popup_map.second) && (*std::get<0>(popup_map.second))) {
            ImGui::OpenPopup(popup_map.first.c_str());
            (*std::get<0>(popup_map.second)) = false;
            // Mirror message in console log with info level
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(std::get<2>(popup_map.second).c_str());
        }
        if (ImGui::BeginPopupModal(popup_map.first.c_str(), nullptr, popup_flags)) {
            ImGui::TextUnformatted(std::get<2>(popup_map.second).c_str());
            bool close = false;
            if (ImGui::Button("Ok")) {
                close = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Ok - Disable further notifications.")) {
                close = true;
                // Disable further notifications
                std::get<1>(popup_map.second) = true;
            }
            if (close || ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape))) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }

    // ABOUT
    if (this->gui_state.open_popup_about) {
        this->gui_state.open_popup_about = false;
        ImGui::OpenPopup("About");
    }
    bool open = true;
    if (ImGui::BeginPopupModal("About", &open, (ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove))) {

        const std::string email("megamol@visus.uni-stuttgart.de");
        const std::string web_link("https://megamol.org/");
        const std::string github_link("https://github.com/UniStuttgart-VISUS/megamol");
        const std::string docu_link("https://github.com/UniStuttgart-VISUS/megamol/tree/master/plugins/gui");
        const std::string imgui_link("https://github.com/ocornut/imgui");

        const std::string mmstr = std::string("MegaMol - Version ") + std::to_string(MEGAMOL_CORE_MAJOR_VER) + (".") +
                                  std::to_string(MEGAMOL_CORE_MINOR_VER) + ("\ngit# ") +
                                  std::string(MEGAMOL_CORE_COMP_REV) + ("\n");
        const std::string mailstr = std::string("Contact: ") + email;
        const std::string webstr = std::string("Web: ") + web_link;
        const std::string gitstr = std::string("Git-Hub: ") + github_link;
        const std::string imguistr = ("Dear ImGui - Version ") + std::string(IMGUI_VERSION) + ("\n");
        const std::string imguigitstr = std::string("Git-Hub: ") + imgui_link;
        const std::string about = "Copyright (C) 2009-2020 by University of Stuttgart (VISUS).\nAll rights reserved.";

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
        if (ImGui::Button("Close") || ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Escape))) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    // Save project pop-up
    if (auto graph_ptr = this->win_configurator_ptr->GetGraphCollection().GetRunningGraph()) {
        this->gui_state.open_popup_save |= this->hotkeys[HOTKEY_GUI_SAVE_PROJECT].is_pressed;
        this->gui_state.open_popup_save |= this->win_configurator_ptr->ConsumeTriggeredGlobalProjectSave();

        auto filename = graph_ptr->GetFilename();
        auto save_gui_state = vislib::math::Ternary(vislib::math::Ternary::TRI_FALSE);
        bool popup_failed = false;
        if (this->file_browser.PopUp_Save(
                "Save Project", {"lua"}, this->gui_state.open_popup_save, filename, save_gui_state)) {
            std::string state_str;
            if (save_gui_state.IsTrue()) {
                state_str = this->project_to_lua_string(true);
            }
            popup_failed = !this->win_configurator_ptr->GetGraphCollection().SaveProjectToFile(
                graph_ptr->UID(), filename, state_str);
        }
        PopUps::Minimal(
            "Failed to Save Project", popup_failed, "See console log output for more information.", "Cancel");
    }
    this->hotkeys[HOTKEY_GUI_SAVE_PROJECT].is_pressed = false;

    // Load project pop-up
    std::string filename;
    this->gui_state.open_popup_load |= this->hotkeys[HOTKEY_GUI_LOAD_PROJECT].is_pressed;
    if (this->file_browser.PopUp_Load("Load Project", {"lua", "png"}, this->gui_state.open_popup_load, filename)) {
        // Redirect project loading request to Lua_Wrapper_service and load new project to megamol graph
        /// GUI graph and GUI state are updated at next synchronization
        this->gui_state.request_load_projet_file = filename;
    }
    this->hotkeys[HOTKEY_GUI_LOAD_PROJECT].is_pressed = false;

    // File name for screenshot pop-up
    auto tmp_flag = vislib::math::Ternary(vislib::math::Ternary::TRI_UNKNOWN);
    if (this->file_browser.PopUp_Save("Filename for Screenshot", {"png"}, this->gui_state.open_popup_screenshot,
            this->gui_state.screenshot_filepath, tmp_flag)) {
        this->gui_state.screenshot_filepath_id = 0;
    }
}


bool megamol::gui::GUIManager::is_hotkey_pressed(megamol::core::view::KeyCode keycode) const {

    ImGuiIO& io = ImGui::GetIO();
    return (ImGui::IsKeyDown(static_cast<int>(keycode.key))) &&
           (keycode.mods.test(core::view::Modifier::ALT) == io.KeyAlt) &&
           (keycode.mods.test(core::view::Modifier::CTRL) == io.KeyCtrl) &&
           (keycode.mods.test(core::view::Modifier::SHIFT) == io.KeyShift);
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
                    state_str, {"font_file_name"}, &this->gui_state.font_load_filename);
                megamol::core::utility::get_json_value<int>(state_str, {"font_size"}, &this->gui_state.font_load_size);
                this->gui_state.font_load = 2;
                std::string imgui_settings;
                megamol::core::utility::get_json_value<std::string>(state_str, {"imgui_settings"}, &imgui_settings);
                this->load_imgui_settings_from_string(imgui_settings);
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

    try {
        out_state.clear();
        nlohmann::json json_state;

        // Write GUI state
        json_state[GUI_JSON_TAG_GUI]["menu_visible"] = this->gui_state.menu_visible;
        json_state[GUI_JSON_TAG_GUI]["style"] = static_cast<int>(this->gui_state.style);
        gui_utils::Utf8Encode(this->gui_state.font_load_filename);
        json_state[GUI_JSON_TAG_GUI]["font_file_name"] = this->gui_state.font_load_filename;
        gui_utils::Utf8Decode(this->gui_state.font_load_filename);
        json_state[GUI_JSON_TAG_GUI]["font_size"] = this->gui_state.font_load_size;
        json_state[GUI_JSON_TAG_GUI]["imgui_settings"] = this->save_imgui_settings_to_string();

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
        do {
            // Create new filename with iterating suffix
            std::string filename = megamol::core::utility::FileUtils::GetFilenameStem<std::string>(inout_filepath);
            std::string id_separator = "_";
            bool new_separator = false;
            auto separator_index = filename.find_last_of(id_separator);
            if (separator_index != std::string::npos) {
                auto last_id_str = filename.substr(separator_index + 1);
                try {
                    this->gui_state.screenshot_filepath_id = std::stoi(last_id_str);
                } catch (...) { new_separator = true; }
                this->gui_state.screenshot_filepath_id++;
                if (new_separator) {
                    this->gui_state.screenshot_filepath =
                        filename + id_separator + std::to_string(this->gui_state.screenshot_filepath_id) + ".png";
                } else {
                    inout_filepath = filename.substr(0, separator_index + 1) +
                                     std::to_string(this->gui_state.screenshot_filepath_id) + ".png";
                }
            } else {
                inout_filepath =
                    filename + id_separator + std::to_string(this->gui_state.screenshot_filepath_id) + ".png";
            }
        } while (megamol::core::utility::FileUtils::FileExists<std::string>(inout_filepath));
        created_filepath = true;
    }
    return created_filepath;
}


void GUIManager::RegisterWindow(
    const std::string& window_name, std::function<void(AbstractWindow::BasicConfig&)> const& callback) {

    this->win_collection.AddWindow(window_name, callback);
}


void GUIManager::RegisterPopUp(const std::string& name, bool& open, const std::function<void()>& callback) {

    this->popup_collection[name] =
        std::pair<bool*, std::function<void()>>(&open, const_cast<std::function<void()>&>(callback));
}


void GUIManager::RegisterNotification(const std::string& name, bool& open, const std::string& message) {

    this->notification_collection[name] = std::tuple<bool*, bool, std::string>(&open, false, message);
}


std::string GUIManager::extract_fontname(const std::string& imgui_fontname) const {

    auto return_fontname = std::string(imgui_fontname);
    auto sep_index = return_fontname.find(',');
    return_fontname = return_fontname.substr(0, sep_index);
    return return_fontname;
}
