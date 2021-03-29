/*
 * GUIWindows.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GUIWindows.h"


using namespace megamol;
using namespace megamol::gui;


GUIWindows::GUIWindows(void)
        : core_instance(nullptr)
        , hotkeys()
        , context(nullptr)
        , initialized_api(megamol::gui::GUIImGuiAPI::NONE)
        , window_collection()
        , configurator()
        , console()
        , state()
        , file_browser()
        , search_widget()
        , tf_editor_ptr(nullptr)
        , tooltip()
        , picking_buffer() {

    this->hotkeys[GUIWindows::GuiHotkeyIndex::TRIGGER_SCREENSHOT] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F2, core::view::Modifier::NONE), false};
    this->hotkeys[GUIWindows::GuiHotkeyIndex::TOGGLE_GRAPH_ENTRY] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F3, core::view::Modifier::NONE), false};
    this->hotkeys[GUIWindows::GuiHotkeyIndex::EXIT_PROGRAM] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F4, core::view::Modifier::ALT), false};
    this->hotkeys[GUIWindows::GuiHotkeyIndex::MENU] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F12, core::view::Modifier::NONE), false};
    this->hotkeys[GUIWindows::GuiHotkeyIndex::PARAMETER_SEARCH] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_P, core::view::Modifier::CTRL), false};
    this->hotkeys[GUIWindows::GuiHotkeyIndex::SAVE_PROJECT] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_S, core::view::Modifier::CTRL), false};
    this->hotkeys[GUIWindows::GuiHotkeyIndex::LOAD_PROJECT] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_L, core::view::Modifier::CTRL), false};
    this->hotkeys[GUIWindows::GuiHotkeyIndex::SHOW_HIDE_GUI] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_G, core::view::Modifier::CTRL), false};

    // Init State
    this->init_state();

    this->tf_editor_ptr = std::make_shared<TransferFunctionEditor>();
    if (this->tf_editor_ptr == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to create transfer function editor. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
}


GUIWindows::~GUIWindows(void) {

    this->destroyContext();
}


bool GUIWindows::CreateContext(GUIImGuiAPI imgui_api, megamol::core::CoreInstance* core_instance) {

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
#else // Linux
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
#endif /// _WIN32
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

    // Set pointer to core instance
    if (core_instance == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Pointer to core instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->core_instance = core_instance;

    // Create ImGui Context
    bool other_context_exists = (ImGui::GetCurrentContext() != nullptr);
    if (this->createContext()) {

        // Initialize ImGui API
        if (!other_context_exists) {
            switch (imgui_api) {
            case (GUIImGuiAPI::OPEN_GL): {
                // Init OpenGL for ImGui
                const char* glsl_version = "#version 150"; /// or "#version 130" or nullptr
                if (ImGui_ImplOpenGL3_Init(glsl_version)) {
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Created ImGui context for Open GL.");
                } else {
                    this->destroyContext();
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Unable to initialize OpenGL for ImGui. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                        __LINE__);
                    return false;
                }

            } break;
            default: {
                this->destroyContext();
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


bool GUIWindows::PreDraw(glm::vec2 framebuffer_size, glm::vec2 window_size, double instance_time) {

    // Handle multiple ImGui contexts.
    if (this->state.gui_visible && ImGui::GetCurrentContext()->WithinFrameScope) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Nesting ImGui contexts is not supported. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        this->state.gui_visible = false;
    }

    // (Delayed font loading for being resource directories available via resource in frontend)
    if (this->state.load_fonts) {
        this->load_default_fonts();
        this->state.load_fonts = false;
    }

    // Process hotkeys
    this->checkMultipleHotkeyAssignement();
    if (this->hotkeys[GUIWindows::GuiHotkeyIndex::SHOW_HIDE_GUI].is_pressed) {
        if (this->state.gui_visible) {
            this->state.gui_hide_next_frame = 2;
        } else { /// !this->state.gui_visible
            // Show GUI after it was hidden (before early exit!)
            // Restore window 'open' state (Always restore at least menu)
            this->state.menu_visible = true;
            const auto func = [&, this](WindowCollection::WindowConfiguration& wc) {
                if (std::find(this->state.gui_visible_buffer.begin(), this->state.gui_visible_buffer.end(),
                        wc.win_callback) != this->state.gui_visible_buffer.end()) {
                    wc.win_show = true;
                }
            };
            this->window_collection.EnumWindows(func);
            this->state.gui_visible_buffer.clear();
            this->state.gui_visible = true;
        }
        this->hotkeys[GUIWindows::GuiHotkeyIndex::SHOW_HIDE_GUI].is_pressed = false;
    }
    if (this->hotkeys[GUIWindows::GuiHotkeyIndex::EXIT_PROGRAM].is_pressed) {
        this->triggerCoreInstanceShutdown();
        this->state.shutdown_triggered = true;
        return true;
    }
    if (this->hotkeys[GUIWindows::GuiHotkeyIndex::TRIGGER_SCREENSHOT].is_pressed) {
        this->state.screenshot_triggered = true;
        this->hotkeys[GUIWindows::GuiHotkeyIndex::TRIGGER_SCREENSHOT].is_pressed = false;
    }
    if (this->hotkeys[GUIWindows::GuiHotkeyIndex::MENU].is_pressed) {
        this->state.menu_visible = !this->state.menu_visible;
        this->hotkeys[GUIWindows::GuiHotkeyIndex::MENU].is_pressed = false;
    }
    if (this->state.toggle_graph_entry || this->hotkeys[GUIWindows::GuiHotkeyIndex::TOGGLE_GRAPH_ENTRY].is_pressed) {
        if (auto graph_ptr = this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid)) {
            megamol::gui::ModulePtrVector_t::const_iterator module_graph_entry_iter = graph_ptr->Modules().begin();
            // Search for first graph entry and set next view to graph entry (= graph entry point)
            for (auto module_iter = graph_ptr->Modules().begin(); module_iter != graph_ptr->Modules().end();
                 module_iter++) {
                if ((*module_iter)->IsView() && (*module_iter)->IsGraphEntry()) {
                    // Remove all graph entries
                    (*module_iter)->SetGraphEntryName("");
                    Graph::QueueData queue_data;
                    queue_data.name_id = (*module_iter)->FullName();
                    graph_ptr->PushSyncQueue(Graph::QueueAction::REMOVE_GRAPH_ENTRY, queue_data);
                    // Save index of last found graph entry
                    if (module_iter != graph_ptr->Modules().end()) {
                        module_graph_entry_iter = module_iter + 1;
                    }
                }
            }
            if ((module_graph_entry_iter == graph_ptr->Modules().begin()) ||
                (module_graph_entry_iter != graph_ptr->Modules().end())) {
                // Search for next graph entry
                for (auto module_iter = module_graph_entry_iter; module_iter != graph_ptr->Modules().end();
                     module_iter++) {
                    if ((*module_iter)->IsView()) {
                        (*module_iter)->SetGraphEntryName(graph_ptr->GenerateUniqueGraphEntryName());
                        Graph::QueueData queue_data;
                        queue_data.name_id = (*module_iter)->FullName();
                        graph_ptr->PushSyncQueue(Graph::QueueAction::CREATE_GRAPH_ENTRY, queue_data);
                        break;
                    }
                }
            }
        }
        this->state.toggle_graph_entry = false;
        this->hotkeys[GUIWindows::GuiHotkeyIndex::TOGGLE_GRAPH_ENTRY].is_pressed = false;
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
    // Propagate ImGui context to core instance
    // if ((this->core_instance != nullptr) && core_instance->IsmmconsoleFrontendCompatible()) { /// mmconsole
    this->core_instance->SetCurrentImGuiContext(this->context);
    //} else {
    /// !!! TODO Move to separate GUI resource which is available in modules
    //}

    // Create new gui graph once if core instance graph is used (otherwise graph should already exist)
    if (this->state.graph_uid == GUI_INVALID_ID) {
        this->SynchronizeGraphs();
    }
    // Check if gui graph is present
    if (this->state.graph_uid == GUI_INVALID_ID) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Failed to find required gui graph for running core graph. [%s, %s, line %d]\n", __FILE__,
            __FUNCTION__, __LINE__);
        return false;
    }

    // Required to prevent change in gui drawing between pre and post draw
    this->state.gui_visible_post = this->state.gui_visible;
    // Early exit when pre step should be omitted
    if (!this->state.gui_visible) {
        return true;
    }

    // Set stuff for next frame --------------------------------------------
    ImGuiIO& io = ImGui::GetIO();

    io.DisplaySize = ImVec2(window_size.x, window_size.y);
    if ((window_size.x > 0.0f) && (window_size.y > 0.0f)) {
        io.DisplayFramebufferScale = ImVec2(framebuffer_size.x / window_size.x, framebuffer_size.y / window_size.y);
    }

    if ((instance_time - this->state.last_instance_time) < 0.0) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Current instance time results in negative time delta. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
    }
    io.DeltaTime = ((instance_time - this->state.last_instance_time) > 0.0)
                       ? (static_cast<float>(instance_time - this->state.last_instance_time))
                       : (io.DeltaTime);
    this->state.last_instance_time = ((instance_time - this->state.last_instance_time) > 0.0)
                                         ? (instance_time)
                                         : (this->state.last_instance_time + io.DeltaTime);

    // Style
    if (this->state.style_changed) {
        ImGuiStyle& style = ImGui::GetStyle();
        switch (this->state.style) {
        case (GUIWindows::Styles::DarkColors): {
            DefaultStyle();
            ImGui::StyleColorsDark();
            style.WindowRounding = 0.0f;
            style.Colors[ImGuiCol_ChildBg] = style.Colors[ImGuiCol_WindowBg];
        } break;
        case (GUIWindows::Styles::LightColors): {
            DefaultStyle();
            ImGui::StyleColorsLight();
            style.WindowRounding = 0.0f;
            style.Colors[ImGuiCol_ChildBg] = style.Colors[ImGuiCol_WindowBg];
        } break;
        case (GUIWindows::Styles::CorporateGray): {
            CorporateGreyStyle();
        } break;
        case (GUIWindows::Styles::CorporateWhite): {
            CorporateWhiteStyle();
        } break;
        default:
            break;
        }
        this->state.style_changed = false;
    }

    // Delete window
    if (!this->state.win_delete.empty()) {
        this->window_collection.DeleteWindowConfiguration(this->state.win_delete);
        this->state.win_delete.clear();
    }

    // Start new ImGui frame --------------------------------------------------
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();

    return true;
}


bool GUIWindows::PostDraw(void) {

    // Early exit when post step should be omitted
    if (!this->state.gui_visible_post) {
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

    ////////// DRAW ///////////////////////////////////////////////////////////

    // Main Menu ---------------------------------------------------------------
    this->drawMenu();

    this->ShowTextures();
    this->ShowHeadnodeRemoteControl();

    // Global Docking Space ---------------------------------------------------
    /// DOCKING
#if (defined(IMGUI_HAS_VIEWPORT) && defined(IMGUI_HAS_DOCK))
    auto child_bg = style.Colors[ImGuiCol_ChildBg];
    style.Colors[ImGuiCol_ChildBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
    style.Colors[ImGuiCol_ChildBg] = child_bg;
#endif

    // Draw Windows ------------------------------------------------------------
    const auto func = [&, this](WindowCollection::WindowConfiguration& wc) {
        // Update transfer function
        if ((wc.win_callback == WindowCollection::DrawCallbacks::TRANSFER_FUNCTION) && wc.buf_tfe_reset) {

            this->tf_editor_ptr->SetMinimized(wc.tfe_view_minimized);
            this->tf_editor_ptr->SetVertical(wc.tfe_view_vertical);

            if (!wc.tfe_active_param.empty()) {
                if (auto graph_ptr = this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid)) {
                    for (auto& module_ptr : graph_ptr->Modules()) {
                        std::string module_full_name = module_ptr->FullName();
                        for (auto& param : module_ptr->Parameters()) {
                            std::string param_full_name = module_full_name + "::" + param.FullName();
                            if ((wc.tfe_active_param == param_full_name) &&
                                (param.Type() == Param_t::TRANSFERFUNCTION)) {
                                this->tf_editor_ptr->SetConnectedParameter(&param, param_full_name);
                                param.TransferFunctionEditor_ConnectExternal(this->tf_editor_ptr, true);
                            }
                        }
                    }
                }
            }
            wc.buf_tfe_reset = false;
        }
        // Update log console
        if (wc.win_callback == WindowCollection::DrawCallbacks::LOGCONSOLE) {
            this->console.Update(wc);
        }
        // Update frame statistics
        if (wc.win_callback == WindowCollection::DrawCallbacks::PERFORMANCE) {
            this->update_frame_statistics(wc);
        }

        // Draw window content
        if (wc.win_show) {

            // Change window flags depending on current view of transfer function editor
            if (wc.win_callback == WindowCollection::DrawCallbacks::TRANSFER_FUNCTION) {
                if (this->tf_editor_ptr->IsMinimized()) {
                    wc.win_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar;

                } else {
                    wc.win_flags = ImGuiWindowFlags_AlwaysAutoResize;
                }
                wc.tfe_view_minimized = this->tf_editor_ptr->IsMinimized();
                wc.tfe_view_vertical = this->tf_editor_ptr->IsVertical();
            }

            ImGui::SetNextWindowBgAlpha(1.0f);
            ImGui::SetNextWindowCollapsed(wc.win_collapsed, ImGuiCond_Always);

            // Begin Window
            auto window_title = wc.win_name + "     " + wc.win_hotkey.ToString();
            if (!ImGui::Begin(window_title.c_str(), &wc.win_show, wc.win_flags)) {
                wc.win_collapsed = ImGui::IsWindowCollapsed();
                ImGui::End(); // early ending
                return;
            }

            // Omit updating size and position of window from imgui for current frame when reset
            bool update_window_by_imgui = !wc.buf_set_pos_size;
            bool collapsing_changed = false;
            this->window_sizing_and_positioning(wc, collapsing_changed);

            // Calling callback drawing window content
            auto cb = this->window_collection.WindowCallback(wc.win_callback);
            if (cb) {
                cb(wc);
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Missing valid callback for WindowDrawCallback: '%d'. [%s, %s, line %d]\n",
                    (int) wc.win_callback, __FILE__, __FUNCTION__, __LINE__);
            }

            // Saving some of the current window state.
            if (update_window_by_imgui) {
                wc.win_position = ImGui::GetWindowPos();
                wc.win_size = ImGui::GetWindowSize();
                if (!collapsing_changed) {
                    wc.win_collapsed = ImGui::IsWindowCollapsed();
                }
            }

            ImGui::End();
        }
    };
    this->window_collection.EnumWindows(func);

    // Draw global parameter widgets -------------------------------------------

    // Enable OpenGL picking
    /// ! Is only enabled in second frame if interaction objects are added during first frame !
    this->picking_buffer.EnableInteraction(glm::vec2(io.DisplaySize.x, io.DisplaySize.y));

    if (auto graph_ptr = this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid)) {
        for (auto& module_ptr : graph_ptr->Modules()) {

            module_ptr->GUIParameterGroups().Draw(module_ptr->Parameters(), module_ptr->FullName(), "",
                vislib::math::Ternary::TRI_UNKNOWN, false, Parameter::WidgetScope::GLOBAL, this->tf_editor_ptr, nullptr,
                GUI_INVALID_ID, &this->picking_buffer);
        }
    }

    // Disable OpenGL picking
    this->picking_buffer.DisableInteraction();

    // Draw pop-ups ------------------------------------------------------------
    this->drawPopUps();

    ///////////////////////////////////////////////////////////////////////////

    // Render the current ImGui frame ------------------------------------------
    glViewport(0, 0, static_cast<GLsizei>(io.DisplaySize.x), static_cast<GLsizei>(io.DisplaySize.y));
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Reset hotkeys ----------------------------------------------------------
    for (auto& h : this->hotkeys) {
        h.is_pressed = false;
    }

    // Hide GUI if it is currently shown --------------------------------------
    if (this->state.gui_visible) {
        if (this->state.gui_hide_next_frame == 2) {
            // First frame
            this->state.gui_hide_next_frame--;
            // Save 'open' state of windows for later restore. Closing all windows before omitting GUI rendering is
            // required to set right ImGui state for mouse handling
            this->state.gui_visible_buffer.clear();
            const auto func = [&, this](WindowCollection::WindowConfiguration& wc) {
                if (wc.win_show) {
                    this->state.gui_visible_buffer.push_back(wc.win_callback);
                    wc.win_show = false;
                }
            };
            this->window_collection.EnumWindows(func);
        } else if (this->state.gui_hide_next_frame == 1) {
            // Second frame
            this->state.gui_hide_next_frame = 0;
            this->state.gui_visible = false;
        }
    }

    // Apply new gui scale -----------------------------------------------------
    if (megamol::gui::gui_scaling.ConsumePendingChange()) {

        // Scale all ImGui style options
        style.ScaleAllSizes(megamol::gui::gui_scaling.TransitionFactor());

        // Scale all windows
        if (this->state.rescale_windows) {
            // Do not adjust window scale after loading from project file (window size is already fine)
            const auto size_func = [&, this](WindowCollection::WindowConfiguration& wc) {
                wc.win_reset_size *= megamol::gui::gui_scaling.TransitionFactor();
                wc.win_size *= megamol::gui::gui_scaling.TransitionFactor();
                wc.buf_set_pos_size = true;
            };
            this->window_collection.EnumWindows(size_func);
            this->state.rescale_windows = false;
        }

        // Reload and scale all fonts
        this->state.load_fonts = true;
    }

    // Loading new font -------------------------------------------------------
    // (after first imgui frame for default fonts being available)
    if (this->state.font_apply) {
        bool load_success = false;
        if (megamol::core::utility::FileUtils::FileWithExtensionExists<std::string>(
                this->state.font_file_name, std::string("ttf"))) {
            ImFontConfig config;
            config.OversampleH = 4;
            config.OversampleV = 4;
            config.GlyphRanges = this->state.font_utf8_ranges.data();
            GUIUtils::Utf8Encode(this->state.font_file_name);
            if (io.Fonts->AddFontFromFileTTF(this->state.font_file_name.c_str(),
                    static_cast<float>(this->state.font_size), &config) != nullptr) {

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
            GUIUtils::Utf8Decode(this->state.font_file_name);
        } else if (this->state.font_file_name != "<unknown>") {
            std::string imgui_font_string =
                this->state.font_file_name + ", " + std::to_string(this->state.font_size) + "px";
            for (unsigned int n = this->state.graph_fonts_reserved; n < static_cast<unsigned int>(io.Fonts->Fonts.Size);
                 n++) {
                std::string font_name = std::string(io.Fonts->Fonts[n]->GetDebugName());
                GUIUtils::Utf8Decode(font_name);
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
        //        this->state.font_file_name.c_str(), this->state.font_size, __FILE__, __FUNCTION__, __LINE__);
        //}
        this->state.font_apply = false;
    }

    return true;
}


bool GUIWindows::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {

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

    // GUI
    for (auto& h : this->hotkeys) {
        if (this->isHotkeyPressed(h.keycode)) {
            h.is_pressed = true;
            hotkeyPressed = true;
        }
    }
    // Configurator
    for (auto& h : this->configurator.GetHotkeys()) {
        if (this->isHotkeyPressed(h.keycode)) {
            h.is_pressed = true;
            hotkeyPressed = true;
        }
    }
    if (hotkeyPressed)
        return true;

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

    // Hotkeys for showing/hiding window(s)
    const auto windows_func = [&](WindowCollection::WindowConfiguration& wc) {
        bool windowHotkeyPressed = this->isHotkeyPressed(wc.win_hotkey);
        if (windowHotkeyPressed) {
            wc.win_show = !wc.win_show;
        }
        hotkeyPressed |= windowHotkeyPressed;
    };
    this->window_collection.EnumWindows(windows_func);
    if (hotkeyPressed)
        return true;

    // Always consume keyboard input if requested by any imgui widget (e.g. text input).
    // User expects hotkey priority of text input thus needs to be processed before parameter hotkeys.
    if (io.WantTextInput) { /// io.WantCaptureKeyboard
        return true;
    }

    // Collect modules which should be considered for parameter hotkey check.
    bool check_all_modules = false;
    std::vector<std::string> modules_list;
    const auto modfunc = [&](WindowCollection::WindowConfiguration& wc) {
        for (auto& m : wc.param_modules_list) {
            modules_list.emplace_back(m);
        }
        if (wc.param_modules_list.empty())
            check_all_modules = true;
    };
    this->window_collection.EnumWindows(modfunc);
    // Check for parameter hotkeys
    hotkeyPressed = false;
    if (auto graph_ptr = this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid)) {
        for (auto& module_ptr : graph_ptr->Modules()) {
            // Break loop after first occurrence of parameter hotkey
            if (hotkeyPressed)
                break;
            if (check_all_modules || this->considerModule(module_ptr->FullName(), modules_list)) {
                for (auto& param : module_ptr->Parameters()) {
                    if (param.Type() == Param_t::BUTTON) {
                        auto keyCode = param.GetStorage<megamol::core::view::KeyCode>();
                        if (this->isHotkeyPressed(keyCode)) {
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
    }

    return hotkeyPressed;
}


bool GUIWindows::OnChar(unsigned int codePoint) {
    ImGui::SetCurrentContext(this->context);

    ImGuiIO& io = ImGui::GetIO();
    io.ClearInputCharacters();
    if (codePoint > 0 && codePoint < 0x10000) {
        io.AddInputCharacter((unsigned short) codePoint);
    }

    return false;
}


bool GUIWindows::OnMouseMove(double x, double y) {
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


bool GUIWindows::OnMouseButton(
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


bool GUIWindows::OnMouseScroll(double dx, double dy) {
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


bool megamol::gui::GUIWindows::GetTriggeredScreenshot(void) {

    bool trigger_screenshot = this->state.screenshot_triggered;
    this->state.screenshot_triggered = false;
    if (trigger_screenshot) {
        this->create_not_existing_png_filepath(this->state.screenshot_filepath);
        if (this->state.screenshot_filepath.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Filename for screenshot should not be empty. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
            trigger_screenshot = false;
        }
    }

    return trigger_screenshot;
}


void megamol::gui::GUIWindows::SetScale(float scale) {
    megamol::gui::gui_scaling.Set(scale);
    if (megamol::gui::gui_scaling.PendingChange()) {
        // Additionally trigger reload of currently used font
        this->state.font_apply = true;
        this->state.font_size = static_cast<int>(
            static_cast<float>(this->state.font_size) * (megamol::gui::gui_scaling.TransitionFactor()));
        // Additionally resize all windows
        this->state.rescale_windows = true;
    }
}


void megamol::gui::GUIWindows::SetClipboardFunc(const char* (*get_clipboard_func)(void* user_data),
    void (*set_clipboard_func)(void* user_data, const char* string), void* user_data) {

    if (this->context != nullptr) {
        ImGuiIO& io = ImGui::GetIO();
        io.SetClipboardTextFn = set_clipboard_func;
        io.GetClipboardTextFn = get_clipboard_func;
        io.ClipboardUserData = user_data;
    }
}


bool megamol::gui::GUIWindows::SynchronizeGraphs(megamol::core::MegaMolGraph* megamol_graph) {

    // 1) Load all known calls from core instance ONCE ---------------------------
    if (!this->configurator.GetGraphCollection().LoadCallStock(core_instance)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Failed to load call stock once. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    // Load all known modules from core instance ONCE
    if (!this->configurator.GetGraphCollection().LoadModuleStock(core_instance)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Failed to load module stock once. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    bool synced = false;
    bool sync_success = false;
    GraphPtr_t graph_ptr = this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid);
    // 2a) Either synchronize GUI Graph -> Core Graph ... ---------------------
    if (!synced && (graph_ptr != nullptr)) {
        bool graph_sync_success = true;

        Graph::QueueAction action;
        Graph::QueueData data;
        while (graph_ptr->PopSyncQueue(action, data)) {
            synced = true;
            switch (action) {
            case (Graph::QueueAction::ADD_MODULE): {
                if (megamol_graph != nullptr) {
                    graph_sync_success &= megamol_graph->CreateModule(data.class_name, data.name_id);
                } else if ((this->core_instance != nullptr) &&
                           core_instance->IsmmconsoleFrontendCompatible()) { /// mmconsole
                    graph_sync_success &= this->core_instance->RequestModuleInstantiation(
                        vislib::StringA(data.class_name.c_str()), vislib::StringA(data.name_id.c_str()));
                }
            } break;
            case (Graph::QueueAction::RENAME_MODULE): {
                if (megamol_graph != nullptr) {
                    bool rename_success = megamol_graph->RenameModule(data.name_id, data.rename_id);
                    graph_sync_success &= rename_success;
                } else if ((this->core_instance != nullptr) &&
                           core_instance->IsmmconsoleFrontendCompatible()) { /// mmconsole
                    /* XXX Currently not supported by core graph
                    bool rename_success = false;
                    std::function<void(megamol::core::Module*)> fun = [&](megamol::core::Module* mod) {
                        mod->setName(vislib::StringA(data.rename_id.c_str()));
                        rename_success = true;
                    };
                    this->core_instance->FindModuleNoLock(data.name_id, fun);
                    graph_sync_success &= rename_success;
                    */
                }
            } break;
            case (Graph::QueueAction::DELETE_MODULE): {
                if (megamol_graph != nullptr) {
                    graph_sync_success &= megamol_graph->DeleteModule(data.name_id);
                } else if (this->core_instance != nullptr) {
                    graph_sync_success &=
                        this->core_instance->RequestModuleDeletion(vislib::StringA(data.name_id.c_str()));
                }
            } break;
            case (Graph::QueueAction::ADD_CALL): {
                if (megamol_graph != nullptr) {
                    graph_sync_success &= megamol_graph->CreateCall(data.class_name, data.caller, data.callee);
                } else if ((this->core_instance != nullptr) &&
                           core_instance->IsmmconsoleFrontendCompatible()) { /// mmconsole
                    graph_sync_success &=
                        this->core_instance->RequestCallInstantiation(vislib::StringA(data.class_name.c_str()),
                            vislib::StringA(data.caller.c_str()), vislib::StringA(data.callee.c_str()));
                }
            } break;
            case (Graph::QueueAction::DELETE_CALL): {
                if (megamol_graph != nullptr) {
                    graph_sync_success &= megamol_graph->DeleteCall(data.caller, data.callee);
                } else if ((this->core_instance != nullptr) &&
                           core_instance->IsmmconsoleFrontendCompatible()) { /// mmconsole
                    graph_sync_success &= this->core_instance->RequestCallDeletion(
                        vislib::StringA(data.caller.c_str()), vislib::StringA(data.callee.c_str()));
                }
            } break;
            case (Graph::QueueAction::CREATE_GRAPH_ENTRY): {
                if (megamol_graph != nullptr) {
                    megamol_graph->SetGraphEntryPoint(data.name_id);
                } else if ((this->core_instance != nullptr) &&
                           core_instance->IsmmconsoleFrontendCompatible()) { /// mmconsole
                    /* XXX Currently not supported by core graph
                     */
                }
            } break;
            case (Graph::QueueAction::REMOVE_GRAPH_ENTRY): {
                if (megamol_graph != nullptr) {
                    megamol_graph->RemoveGraphEntryPoint(data.name_id);
                } else if ((this->core_instance != nullptr) &&
                           core_instance->IsmmconsoleFrontendCompatible()) { /// mmconsole
                    /* XXX Currently not supported by core graph
                     */
                }
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
        bool graph_sync_success = this->configurator.GetGraphCollection().LoadUpdateProjectFromCore(
            this->state.graph_uid, this->core_instance, megamol_graph, true);
        if (!graph_sync_success) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Failed to synchronize core graph with gui graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
        }

        // Check for new GUI state
        if (!this->state.new_gui_state.empty()) {
            this->state_from_string(this->state.new_gui_state);
            this->state.new_gui_state.clear();
        }

        // Check for new script path name
        if (graph_sync_success) {
            if (auto graph_ptr = this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid)) {
                std::string script_filename;
                // Get project filename from lua state of core instance
                if ((this->core_instance != nullptr) && core_instance->IsmmconsoleFrontendCompatible()) { /// mmconsole
                    if (auto lua_state = this->core_instance->GetLuaState()) {
                        script_filename = lua_state->GetScriptPath();
                    }
                } else {
                    // Get project filename from lua state of frontend service
                    if (!this->state.project_script_paths.empty()) {
                        script_filename = this->state.project_script_paths.front();
                    }
                }
                // Load GUI state from project file when project file changed
                if (!script_filename.empty()) {
                    graph_ptr->SetFilename(script_filename);
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
                    if (megamol_graph != nullptr) {
                        core_module_ptr = megamol_graph->FindModule(module_name).get();
                    } else if ((this->core_instance != nullptr) &&
                               core_instance->IsmmconsoleFrontendCompatible()) { /// mmconsole
                        // New core module will only be available next frame after module request is processed.
                        std::function<void(megamol::core::Module*)> fun = [&](megamol::core::Module* mod) {
                            core_module_ptr = mod;
                        };
                        this->core_instance->FindModuleNoLock(module_name, fun);
                    }
                    // Connect pointer of new parameters of core module to parameters in gui module
                    if (core_module_ptr != nullptr) {
                        megamol::core::AbstractNamedObjectContainer::child_list_type::const_iterator se =
                            core_module_ptr->ChildList_End();
                        for (megamol::core::AbstractNamedObjectContainer::child_list_type::const_iterator si =
                                 core_module_ptr->ChildList_Begin();
                             si != se; ++si) {
                            auto param_slot = dynamic_cast<megamol::core::param::ParamSlot*>((*si).get());
                            if (param_slot != nullptr) {
                                std::string param_full_name(param_slot->Name().PeekBuffer());
                                for (auto& parameter : module_ptr->Parameters()) {
                                    if (parameter.FullName() == param_full_name) {
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


bool GUIWindows::createContext(void) {

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

    // Register window callbacks in window collection -------------------------
    this->window_collection.RegisterDrawWindowCallback(WindowCollection::DrawCallbacks::MAIN_PARAMETERS,
        [&, this](WindowCollection::WindowConfiguration& wc) { this->drawParamWindowCallback(wc); });
    this->window_collection.RegisterDrawWindowCallback(WindowCollection::DrawCallbacks::PARAMETERS,
        [&, this](WindowCollection::WindowConfiguration& wc) { this->drawParamWindowCallback(wc); });
    this->window_collection.RegisterDrawWindowCallback(WindowCollection::DrawCallbacks::PERFORMANCE,
        [&, this](WindowCollection::WindowConfiguration& wc) { this->drawFpsWindowCallback(wc); });
    this->window_collection.RegisterDrawWindowCallback(WindowCollection::DrawCallbacks::TRANSFER_FUNCTION,
        [&, this](WindowCollection::WindowConfiguration& wc) { this->drawTransferFunctionWindowCallback(wc); });
    this->window_collection.RegisterDrawWindowCallback(WindowCollection::DrawCallbacks::CONFIGURATOR,
        [&, this](WindowCollection::WindowConfiguration& wc) { this->drawConfiguratorWindowCallback(wc); });
    this->window_collection.RegisterDrawWindowCallback(WindowCollection::DrawCallbacks::LOGCONSOLE,
        [&, this](WindowCollection::WindowConfiguration& wc) { this->console.Draw(wc); });

    // Create window configurations
    WindowCollection::WindowConfiguration buf_win;
    buf_win.buf_set_pos_size = true;
    buf_win.win_collapsed = false;
    buf_win.win_store_config = true;

    float vp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    ::glGetFloatv(GL_VIEWPORT, vp);

    // CONFIGURATOR Window ----------------------------------------------------
    buf_win.win_name = "Configurator";
    buf_win.win_show = false;
    buf_win.win_size = ImVec2(vp[2], vp[3]);
    buf_win.win_reset_size = buf_win.win_size;
    buf_win.win_position = ImVec2(0.0f, 0.0f);
    buf_win.win_reset_position = buf_win.win_position;
    buf_win.win_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoScrollbar;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F11);
    buf_win.win_callback = WindowCollection::DrawCallbacks::CONFIGURATOR;
    this->window_collection.AddWindowConfiguration(buf_win);

    // Parameters -------------------------------------------------------------
    buf_win.win_name = "Parameters";
    buf_win.win_show = true;
    buf_win.win_size = ImVec2(400.0f, 500.0f);
    buf_win.win_reset_size = buf_win.win_size;
    buf_win.win_position = ImVec2(0.0f, 0.0f);
    buf_win.win_reset_position = buf_win.win_position;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F10);
    buf_win.win_flags = ImGuiWindowFlags_NoScrollbar;
    buf_win.win_callback = WindowCollection::DrawCallbacks::MAIN_PARAMETERS;
    buf_win.win_reset_size = buf_win.win_size;
    this->window_collection.AddWindowConfiguration(buf_win);
    float param_win_width = buf_win.win_size.x;
    float param_win_height = buf_win.win_size.y;

    // LOG CONSOLE Window -----------------------------------------------------
    const float default_font_size = (12.0f * megamol::gui::gui_scaling.Get() + ImGui::GetFrameHeightWithSpacing());
    buf_win.win_name = "Log Console";
    buf_win.win_show = false;
    buf_win.win_size =
        ImVec2(vp[2], std::min((vp[3] - param_win_height - default_font_size), (8.0f * default_font_size)));
    buf_win.win_reset_size = buf_win.win_size;
    buf_win.win_position = ImVec2(0.0f, vp[3] - buf_win.win_size.y);
    buf_win.win_reset_position = buf_win.win_position;
    buf_win.win_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_HorizontalScrollbar;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F9);
    buf_win.win_callback = WindowCollection::DrawCallbacks::LOGCONSOLE;
    this->window_collection.AddWindowConfiguration(buf_win);

    // TRANSFER FUNCTION Window -----------------------------------------------
    buf_win.win_name = "Transfer Function Editor";
    buf_win.win_show = false;
    buf_win.win_size = ImVec2(0.0f, 0.0f); /// see ImGuiWindowFlags_AlwaysAutoResize
    buf_win.win_reset_size = buf_win.win_size;
    buf_win.win_position = ImVec2(param_win_width, 0.0f);
    buf_win.win_reset_position = buf_win.win_position;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F8);
    buf_win.win_flags = ImGuiWindowFlags_AlwaysAutoResize;
    buf_win.win_callback = WindowCollection::DrawCallbacks::TRANSFER_FUNCTION;
    this->window_collection.AddWindowConfiguration(buf_win);

    // FPS/MS Window ----------------------------------------------------------
    buf_win.win_name = "Performance Metrics";
    buf_win.win_show = false;
    buf_win.win_size = ImVec2(0.0f, 0.0f); /// see ImGuiWindowFlags_AlwaysAutoResize
    buf_win.win_reset_size = buf_win.win_size;
    buf_win.win_position = ImVec2(vp[2] / 2.0f, 0.0f);
    buf_win.win_reset_position = buf_win.win_position;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F7);
    buf_win.win_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
    buf_win.win_callback = WindowCollection::DrawCallbacks::PERFORMANCE;
    this->window_collection.AddWindowConfiguration(buf_win);

    // Style settings ---------------------------------------------------------
    ImGui::SetColorEditOptions(ImGuiColorEditFlags_Uint8 | ImGuiColorEditFlags_DisplayRGB |
                               ImGuiColorEditFlags_PickerHueWheel | ImGuiColorEditFlags_AlphaBar |
                               ImGuiColorEditFlags_AlphaPreview);

    // IO settings ------------------------------------------------------------
    ImGuiIO& io = ImGui::GetIO();
    io.IniSavingRate = 5.0f;                              //  in seconds
    io.IniFilename = nullptr;                             // "imgui.ini" - disabled, using own window settings profile
    io.LogFilename = nullptr;                             // "imgui_log.txt" - disabled
    io.FontAllowUserScaling = false;                      // disable font scaling using ctrl + mouse wheel
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // allow keyboard navigation

/// DOCKING
#if (defined(IMGUI_HAS_VIEWPORT) && defined(IMGUI_HAS_DOCK))
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable; // enable window docking
    io.ConfigDockingWithShift = true;                 // activate docking on pressing 'shift'
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
    this->state.font_utf8_ranges.clear();
    this->state.font_utf8_ranges.emplace_back(0x0020);
    this->state.font_utf8_ranges.emplace_back(0x03FF); // Basic Latin + Latin Supplement + Greek Alphabet
    this->state.font_utf8_ranges.emplace_back(0x20AC);
    this->state.font_utf8_ranges.emplace_back(0x20AC); // Euro
    this->state.font_utf8_ranges.emplace_back(0x2122);
    this->state.font_utf8_ranges.emplace_back(0x2122); // TM
    this->state.font_utf8_ranges.emplace_back(0x212B);
    this->state.font_utf8_ranges.emplace_back(0x212B); // Angstroem
    this->state.font_utf8_ranges.emplace_back(0x0391);
    this->state.font_utf8_ranges.emplace_back(0); // (range termination)

    // Load initial fonts only once for all imgui contexts --------------------
    if (other_context_exists) {

        // Fonts are already loaded
        if (default_font != nullptr) {
            io.FontDefault = default_font;
        } else {
            // ... else default font is font loaded after configurator fonts -> Index equals number of graph fonts.
            auto default_font_index = static_cast<int>(this->configurator.GetGraphFontScalings().size());
            default_font_index = std::min(default_font_index, io.Fonts->Fonts.Size - 1);
            io.FontDefault = io.Fonts->Fonts[default_font_index];
        }

    } else {
        this->state.load_fonts = true;
    }

    return true;
}


bool GUIWindows::destroyContext(void) {

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

    this->window_collection.DeleteWindowConfigurations();

    this->configurator.GetGraphCollection().DeleteGraph(this->state.graph_uid);
    this->state.graph_uid = GUI_INVALID_ID;

    this->core_instance = nullptr;

    return true;
}


void megamol::gui::GUIWindows::load_default_fonts(void) {

    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->Clear();

    const auto graph_font_scalings = this->configurator.GetGraphFontScalings();
    this->state.graph_fonts_reserved = graph_font_scalings.size();

    const float default_font_size = (12.0f * megamol::gui::gui_scaling.Get());
    ImFontConfig config;
    config.OversampleH = 4;
    config.OversampleV = 4;
    config.GlyphRanges = this->state.font_utf8_ranges.data();

    // Get other known fonts
    std::vector<std::string> font_paths;
    std::string configurator_font_path;
    std::string default_font_path;

    auto get_preset_font_path = [&](auto directory) {
        std::string font_path = megamol::core::utility::FileUtils::SearchFileRecursive(directory, "Roboto-Regular.ttf");
        if (!font_path.empty()) {
            font_paths.emplace_back(font_path);
            configurator_font_path = font_path;
            default_font_path = font_path;
        }
        font_path = megamol::core::utility::FileUtils::SearchFileRecursive(directory, "SourceCodePro-Regular.ttf");
        if (!font_path.empty()) {
            font_paths.emplace_back(font_path);
        }
    };

    if ((this->core_instance != nullptr) && core_instance->IsmmconsoleFrontendCompatible()) { /// mmconsole
        auto search_paths = this->core_instance->Configuration().ResourceDirectories();
        for (size_t i = 0; i < search_paths.Count(); ++i) {
            get_preset_font_path(std::wstring(search_paths[i].PeekBuffer()));
        }
    } else {
        for (auto& resource_directory : this->state.resource_directories) {
            get_preset_font_path(resource_directory);
        }
    }

    // Configurator Graph Font: Add default font at first n indices for exclusive use in configurator graph.
    /// Workaround: Using different font sizes for different graph zooming factors to improve font readability when
    /// zooming.
    if (configurator_font_path.empty()) {
        for (unsigned int i = 0; i < this->state.graph_fonts_reserved; i++) {
            io.Fonts->AddFontDefault(&config);
        }
    } else {
        for (unsigned int i = 0; i < this->state.graph_fonts_reserved; i++) {
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


void GUIWindows::drawTransferFunctionWindowCallback(WindowCollection::WindowConfiguration& wc) {

    this->tf_editor_ptr->Widget(true);
    wc.tfe_active_param = this->tf_editor_ptr->GetConnectedParameterName();
}


void GUIWindows::drawConfiguratorWindowCallback(WindowCollection::WindowConfiguration& wc) {

    this->configurator.Draw(wc);
}


void GUIWindows::drawParamWindowCallback(WindowCollection::WindowConfiguration& wc) {

    // Mode
    megamol::gui::ButtonWidgets::ExtendedModeButton("draw_param_window_callback", wc.param_extended_mode);
    this->tooltip.Marker("Expert mode enables options for additional parameter presentation options.");
    ImGui::SameLine();

    // Options
    ImGuiID override_header_state = GUI_INVALID_ID;
    if (ImGui::Button("Expand All")) {
        override_header_state = 1; // open
    }
    ImGui::SameLine();
    if (ImGui::Button("Collapse All")) {
        override_header_state = 0; // close
    }
    ImGui::SameLine();

    // Info
    std::string help_marker = "[INFO]";
    std::string param_help = "[Hover] Show Parameter Description Tooltip\n"
                             "[Right Click] Context Menu\n"
                             "[Drag & Drop] Move Module to other Parameter Window\n"
                             "[Enter], [Tab], [Left Click outside Widget] Confirm input changes";
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled(help_marker.c_str());
    this->tooltip.ToolTip(param_help);

    // Paramter substring name filtering (only for main parameter view)
    if (wc.win_callback == WindowCollection::DrawCallbacks::MAIN_PARAMETERS) {
        if (this->hotkeys[GUIWindows::GuiHotkeyIndex::PARAMETER_SEARCH].is_pressed) {
            this->search_widget.SetSearchFocus(true);
            this->hotkeys[GUIWindows::GuiHotkeyIndex::PARAMETER_SEARCH].is_pressed = false;
        }
        std::string help_test = "[" + this->hotkeys[GUIWindows::GuiHotkeyIndex::PARAMETER_SEARCH].keycode.ToString() +
                                "] Set keyboard focus to search input field.\n"
                                "Case insensitive substring search in module and parameter names.\nSearches globally "
                                "in all parameter windows.\n";
        this->search_widget.Widget("guiwindow_parameter_earch", help_test);
    }

    ImGui::Separator();

    // Create child window for sepearte scroll bar and keeping header always visible on top of parameter list
    ImGui::BeginChild("###ParameterList", ImVec2(0.0f, 0.0f), false, ImGuiWindowFlags_HorizontalScrollbar);

    // Listing modules and their parameters
    const size_t dnd_size = 2048; // Set same max size of all module labels for drag and drop.
    if (auto graph_ptr = this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid)) {

        // Get module groups
        std::map<std::string, std::vector<ModulePtr_t>> group_map;
        for (auto& module_ptr : graph_ptr->Modules()) {
            auto group_name = module_ptr->GroupName();
            if (!group_name.empty()) {
                group_map["::" + group_name].emplace_back(module_ptr);
            } else {
                group_map[""].emplace_back(module_ptr);
            }
        }
        for (auto& group : group_map) {
            std::string search_string = this->search_widget.GetSearchString();
            bool indent = false;
            bool group_header_open = group.first.empty();
            if (!group_header_open) {
                group_header_open = GUIUtils::GroupHeader(
                    megamol::gui::HeaderType::MODULE_GROUP, group.first, search_string, override_header_state);
                indent = true;
                ImGui::Indent();
            }
            if (group_header_open) {
                for (auto& module_ptr : group.second) {
                    std::string module_label = module_ptr->FullName();
                    ImGui::PushID(module_ptr->UID());

                    // Check if module should be considered.
                    if (!this->considerModule(module_label, wc.param_modules_list)) {
                        continue;
                    }

                    // Draw module header
                    bool module_header_open = GUIUtils::GroupHeader(
                        megamol::gui::HeaderType::MODULE, module_label, search_string, override_header_state);
                    // Module description as hover tooltip
                    this->tooltip.ToolTip(module_ptr->Description(), ImGui::GetID(module_label.c_str()), 0.5f, 5.0f);

                    // Context menu
                    if (ImGui::BeginPopupContextItem()) {
                        if (ImGui::MenuItem("Copy to new Window")) {
                            std::srand(std::time(nullptr));
                            std::string window_name = "Parameters###parameters_" + std::to_string(std::rand());
                            WindowCollection::WindowConfiguration buf_win;
                            buf_win.win_name = window_name;
                            buf_win.win_show = true;
                            buf_win.win_flags = ImGuiWindowFlags_NoScrollbar;
                            buf_win.win_callback = WindowCollection::DrawCallbacks::PARAMETERS;
                            buf_win.param_show_hotkeys = false;
                            buf_win.win_position =
                                ImVec2(ImGui::GetTextLineHeightWithSpacing(), ImGui::GetTextLineHeightWithSpacing());
                            buf_win.win_size = ImVec2(
                                (400.0f * megamol::gui::gui_scaling.Get()), (600.0f * megamol::gui::gui_scaling.Get()));
                            buf_win.param_modules_list.emplace_back(module_label);
                            this->window_collection.AddWindowConfiguration(buf_win);
                        }

                        // Deleting module's parameters is not available in main parameter window.
                        if (wc.win_callback != WindowCollection::DrawCallbacks::MAIN_PARAMETERS) {
                            if (ImGui::MenuItem("Delete from List")) {
                                std::vector<std::string>::iterator find_iter =
                                    std::find(wc.param_modules_list.begin(), wc.param_modules_list.end(), module_label);
                                // Break if module name is not contained in list
                                if (find_iter != wc.param_modules_list.end()) {
                                    wc.param_modules_list.erase(find_iter);
                                }
                                if (wc.param_modules_list.empty()) {
                                    this->state.win_delete = wc.win_name;
                                }
                            }
                        }
                        ImGui::EndPopup();
                    }

                    // Drag source
                    module_label.resize(dnd_size);
                    if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
                        ImGui::SetDragDropPayload(
                            "DND_COPY_MODULE_PARAMETERS", module_label.c_str(), (module_label.size() * sizeof(char)));
                        ImGui::TextUnformatted(module_label.c_str());
                        ImGui::EndDragDropSource();
                    }

                    // Draw parameters
                    if (module_header_open) {
                        bool out_open_external_tf_editor;
                        module_ptr->GUIParameterGroups().Draw(module_ptr->Parameters(), module_label, search_string,
                            vislib::math::Ternary(wc.param_extended_mode), true, Parameter::WidgetScope::LOCAL,
                            this->tf_editor_ptr, &out_open_external_tf_editor, override_header_state, nullptr);
                        if (out_open_external_tf_editor) {
                            const auto func = [](WindowCollection::WindowConfiguration& wc) {
                                if (wc.win_callback == WindowCollection::DrawCallbacks::TRANSFER_FUNCTION) {
                                    wc.win_show = true;
                                }
                            };
                            this->window_collection.EnumWindows(func);
                        }
                    }

                    ImGui::PopID();
                }
            }
            if (indent) {
                ImGui::Unindent();
            }
        }
    }

    // Drop target
    ImGui::Dummy(ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetFontSize()));
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_COPY_MODULE_PARAMETERS")) {

            IM_ASSERT(payload->DataSize == (dnd_size * sizeof(char)));
            std::string payload_id = (const char*) payload->Data;

            // Insert dragged module name only if not contained in list
            if (!this->considerModule(payload_id, wc.param_modules_list)) {
                wc.param_modules_list.emplace_back(payload_id);
            }
        }
        ImGui::EndDragDropTarget();
    }

    ImGui::EndChild();
}


void GUIWindows::drawFpsWindowCallback(WindowCollection::WindowConfiguration& wc) {

    ImGuiStyle& style = ImGui::GetStyle();

    if (ImGui::RadioButton("fps", (wc.fpsms_mode == WindowCollection::TimingModes::FPS))) {
        wc.fpsms_mode = WindowCollection::TimingModes::FPS;
    }
    ImGui::SameLine();

    if (ImGui::RadioButton("ms", (wc.fpsms_mode == WindowCollection::TimingModes::MS))) {
        wc.fpsms_mode = WindowCollection::TimingModes::MS;
    }

    ImGui::TextDisabled("Frame ID:");
    ImGui::SameLine();
    auto frameid = this->state.stat_frame_count;
    if ((this->core_instance != nullptr) && core_instance->IsmmconsoleFrontendCompatible()) { /// mmconsole
        if (frameid == 0) {
            frameid = static_cast<size_t>(this->core_instance->GetFrameID());
        }
    }
    ImGui::Text("%u", frameid);

    ImGui::SameLine(
        ImGui::CalcItemWidth() - (ImGui::GetFrameHeightWithSpacing() - style.ItemSpacing.x - style.ItemInnerSpacing.x));
    if (ImGui::ArrowButton("Options_", ((wc.fpsms_show_options) ? (ImGuiDir_Down) : (ImGuiDir_Up)))) {
        wc.fpsms_show_options = !wc.fpsms_show_options;
    }

    auto* value_buffer =
        ((wc.fpsms_mode == WindowCollection::TimingModes::FPS) ? (&wc.buf_fps_values) : (&wc.buf_ms_values));
    int buffer_size = static_cast<int>(value_buffer->size());

    std::string value_string;
    if (buffer_size > 0) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(3) << value_buffer->back();
        value_string = stream.str();
    }

    float* value_ptr = value_buffer->data();
    float max_value = ((wc.fpsms_mode == WindowCollection::TimingModes::FPS) ? (wc.buf_fps_max) : (wc.buf_ms_max));
    ImGui::PlotLines("###msplot", value_ptr, buffer_size, 0, value_string.c_str(), 0.0f, (1.5f * max_value),
        ImVec2(0.0f, (50.0f * megamol::gui::gui_scaling.Get())));

    if (wc.fpsms_show_options) {
        if (ImGui::InputFloat("Refresh Rate (per sec.)", &wc.fpsms_refresh_rate, 1.0f, 10.0f, "%.3f",
                ImGuiInputTextFlags_EnterReturnsTrue)) {
            wc.fpsms_refresh_rate = std::max(1.0f, wc.fpsms_refresh_rate);
        }

        if (ImGui::InputInt("History Size", &wc.fpsms_buffer_size, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue)) {
            wc.fpsms_buffer_size = std::max(1, wc.fpsms_buffer_size);
        }

        if (ImGui::Button("Current Value")) {
            ImGui::SetClipboardText(value_string.c_str());
        }
        ImGui::SameLine();

        if (ImGui::Button("All Values")) {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(3);
            auto reverse_end = value_buffer->rend();
            for (std::vector<float>::reverse_iterator i = value_buffer->rbegin(); i != reverse_end; ++i) {
                stream << (*i) << "\n";
            }
            ImGui::SetClipboardText(stream.str().c_str());
        }
        ImGui::SameLine();
        ImGui::TextUnformatted("Copy to Clipborad");
        std::string help("Values are listed in chronological order (newest first).");
        this->tooltip.Marker(help);
    }
}


void GUIWindows::drawMenu(void) {

    if (!this->state.menu_visible)
        return;

    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    bool megamolgraph_interface = false;
    if (auto graph_ptr = this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid)) {
        megamolgraph_interface = (graph_ptr->GetCoreInterface() == GraphCoreInterface::MEGAMOL_GRAPH);
    }

    ImGui::BeginMainMenuBar();

    // FILE -------------------------------------------------------------------
    if (ImGui::BeginMenu("File")) {

        if (megamolgraph_interface) {
            if (ImGui::MenuItem("Load Project",
                    this->hotkeys[GUIWindows::GuiHotkeyIndex::LOAD_PROJECT].keycode.ToString().c_str())) {
                this->state.open_popup_load = true;
            }
        }
        if (ImGui::MenuItem(
                "Save Project", this->hotkeys[GUIWindows::GuiHotkeyIndex::SAVE_PROJECT].keycode.ToString().c_str())) {
            this->state.open_popup_save = true;
        }
        if (ImGui::MenuItem(
                "Exit", this->hotkeys[GUIWindows::GuiHotkeyIndex::EXIT_PROGRAM].keycode.ToString().c_str())) {
            this->triggerCoreInstanceShutdown();
            this->state.shutdown_triggered = true;
        }
        ImGui::EndMenu();
    }
    ImGui::Separator();

    // WINDOWS ----------------------------------------------------------------
    if (ImGui::BeginMenu("Windows")) {
        ImGui::MenuItem("Menu", this->hotkeys[GUIWindows::GuiHotkeyIndex::MENU].keycode.ToString().c_str(),
            &this->state.menu_visible);
        const auto func = [&, this](WindowCollection::WindowConfiguration& wc) {
            bool registered_window = !(wc.win_hotkey.key == core::view::Key::KEY_UNKNOWN);
            if (registered_window) {
                ;
                ImGui::MenuItem(wc.win_name.c_str(), wc.win_hotkey.ToString().c_str(), &wc.win_show);
            } else {
                if (ImGui::BeginMenu(wc.win_name.c_str())) {
                    std::string menu_label = "Show";
                    if (wc.win_show)
                        menu_label = "Hide";
                    if (ImGui::MenuItem(menu_label.c_str(), wc.win_hotkey.ToString().c_str(), nullptr)) {
                        wc.win_show = !wc.win_show;
                    }
                    // Enable option to delete window if it is a newly created custom parameter window
                    if (ImGui::MenuItem("Delete Window")) {
                        this->state.win_delete = wc.win_name;
                    }
                    ImGui::EndMenu();
                }
            }
        };
        this->window_collection.EnumWindows(func);

        ImGui::EndMenu();
    }
    ImGui::Separator();

    // SCREENSHOT -------------------------------------------------------------
    if (megamolgraph_interface) {
        if (ImGui::BeginMenu("Screenshot")) {
            this->create_not_existing_png_filepath(this->state.screenshot_filepath);
            if (ImGui::MenuItem("Select Filename", this->state.screenshot_filepath.c_str())) {
                this->state.open_popup_screenshot = true;
            }
            if (ImGui::MenuItem("Trigger",
                    this->hotkeys[GUIWindows::GuiHotkeyIndex::TRIGGER_SCREENSHOT].keycode.ToString().c_str())) {
                this->state.screenshot_triggered = true;
            }
            ImGui::EndMenu();
        }
        ImGui::Separator();
    }

    // RENDER -----------------------------------------------------------------
    if (megamolgraph_interface) {
        if (auto graph_ptr = this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid)) {
            if (ImGui::BeginMenu("Render")) {
                for (auto& module_ptr : graph_ptr->Modules()) {
                    if (module_ptr->IsView()) {
                        if (ImGui::MenuItem(module_ptr->FullName().c_str(), "", module_ptr->IsGraphEntry())) {
                            if (!module_ptr->IsGraphEntry()) {
                                // Remove all graph entries
                                for (auto module_ptr : graph_ptr->Modules()) {
                                    if (module_ptr->IsView() && module_ptr->IsGraphEntry()) {
                                        module_ptr->SetGraphEntryName("");
                                        Graph::QueueData queue_data;
                                        queue_data.name_id = module_ptr->FullName();
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
                        this->hotkeys[GUIWindows::GuiHotkeyIndex::TOGGLE_GRAPH_ENTRY].keycode.ToString().c_str())) {
                    this->state.toggle_graph_entry = true;
                }
                ImGui::EndMenu();
            }
            ImGui::Separator();
        }
    }

    // SETTINGS ---------------------------------------------------------------
    if (ImGui::BeginMenu("Settings")) {

        if (ImGui::MenuItem(
                "Show/Hide GUI", this->hotkeys[GUIWindows::GuiHotkeyIndex::SHOW_HIDE_GUI].keycode.ToString().c_str())) {
            this->state.gui_hide_next_frame = 2;
        }

        if (ImGui::BeginMenu("Style")) {

            if (ImGui::MenuItem("ImGui Dark Colors", nullptr, (this->state.style == GUIWindows::Styles::DarkColors))) {
                this->state.style = GUIWindows::Styles::DarkColors;
                this->state.style_changed = true;
            }
            if (ImGui::MenuItem("ImGui LightColors", nullptr, (this->state.style == GUIWindows::Styles::LightColors))) {
                this->state.style = GUIWindows::Styles::LightColors;
                this->state.style_changed = true;
            }
            if (ImGui::MenuItem("Corporate Gray", nullptr, (this->state.style == GUIWindows::Styles::CorporateGray))) {
                this->state.style = GUIWindows::Styles::CorporateGray;
                this->state.style_changed = true;
            }
            if (ImGui::MenuItem(
                    "Corporate White", nullptr, (this->state.style == GUIWindows::Styles::CorporateWhite))) {
                this->state.style = GUIWindows::Styles::CorporateWhite;
                this->state.style_changed = true;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Font")) {

            ImGuiIO& io = ImGui::GetIO();
            ImFont* font_current = ImGui::GetFont();
            if (ImGui::BeginCombo("Select Available Font", font_current->GetDebugName())) {
                /// first fonts until index this->graph_fonts_reserved are exclusively used by graph in configurator
                for (int n = this->state.graph_fonts_reserved; n < io.Fonts->Fonts.Size; n++) {
                    if (ImGui::Selectable(io.Fonts->Fonts[n]->GetDebugName(), (io.Fonts->Fonts[n] == font_current))) {
                        io.FontDefault = io.Fonts->Fonts[n];
                        // Saving font to window configuration (Remove font size from font name)
                        this->state.font_file_name = std::string(io.FontDefault->GetDebugName());
                        auto sep_index = this->state.font_file_name.find(",");
                        this->state.font_file_name = this->state.font_file_name.substr(0, sep_index);
                        this->state.font_size = static_cast<int>(io.FontDefault->FontSize);
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
            ImGui::InputInt(label.c_str(), &this->state.font_size, 1, 10, ImGuiInputTextFlags_None);
            // Validate font size
            if (this->state.font_size <= 5) {
                this->state.font_size = 5; // minimum valid font size
            }

            ImGui::BeginGroup();
            float widget_width = ImGui::CalcItemWidth() - (ImGui::GetFrameHeightWithSpacing() + style.ItemSpacing.x);
            ImGui::PushItemWidth(widget_width);
            this->file_browser.Button(
                this->state.font_file_name, megamol::gui::FileBrowserWidget::FileBrowserFlag::LOAD, "ttf");
            ImGui::SameLine();
            GUIUtils::Utf8Encode(this->state.font_file_name);
            ImGui::InputText("Font Filename (.ttf)", &this->state.font_file_name, ImGuiInputTextFlags_None);
            GUIUtils::Utf8Decode(this->state.font_file_name);
            ImGui::PopItemWidth();
            // Validate font file before offering load button
            bool valid_file = megamol::core::utility::FileUtils::FileWithExtensionExists<std::string>(
                this->state.font_file_name, std::string("ttf"));
            if (!valid_file) {
                megamol::gui::GUIUtils::ReadOnlyWigetStyle(true);
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            }
            if (ImGui::Button("Add Font")) {
                this->state.font_apply = true;
            }
            if (!valid_file) {
                ImGui::PopItemFlag();
                megamol::gui::GUIUtils::ReadOnlyWigetStyle(false);
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
            this->state.open_popup_about = true;
        }
        ImGui::EndMenu();
    }
    ImGui::Separator();

    ImGui::EndMainMenuBar();
}


void megamol::gui::GUIWindows::drawPopUps(void) {

    // ABOUT
    if (this->state.open_popup_about) {
        this->state.open_popup_about = false;
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
    this->state.open_popup_save |= this->hotkeys[GUIWindows::GuiHotkeyIndex::SAVE_PROJECT].is_pressed;
    bool confirmed, aborted;
    bool popup_failed = false;
    std::string filename;
    GraphPtr_t graph_ptr = this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid);
    if (graph_ptr != nullptr) {
        filename = graph_ptr->GetFilename();
        vislib::math::Ternary save_gui_state(
            vislib::math::Ternary::TRI_FALSE); // Default for option asking for saving gui state
        this->state.open_popup_save |= this->configurator.ConsumeTriggeredGlobalProjectSave();

        if (this->file_browser.PopUp(filename, FileBrowserWidget::FileBrowserFlag::SAVE, "Save Project",
                this->state.open_popup_save, "lua", save_gui_state)) {

            graph_ptr->SetFilename(filename);

            std::string gui_state;
            if (save_gui_state.IsTrue()) {
                gui_state = this->project_to_lua_string();
            }

            popup_failed |=
                !this->configurator.GetGraphCollection().SaveProjectToFile(this->state.graph_uid, filename, gui_state);
        }
        MinimalPopUp::PopUp("Failed to Save Project", popup_failed, "See console log output for more information.", "",
            confirmed, "Cancel", aborted);
    }
    this->state.open_popup_save = false;
    this->hotkeys[GUIWindows::GuiHotkeyIndex::SAVE_PROJECT].is_pressed = false;

    // Load project pop-up
    popup_failed = false;
    if (graph_ptr != nullptr) {
        this->state.open_popup_load |= this->hotkeys[GUIWindows::GuiHotkeyIndex::LOAD_PROJECT].is_pressed;
        if (this->file_browser.PopUp(filename, FileBrowserWidget::FileBrowserFlag::LOAD, "Load Project",
                this->state.open_popup_load, "lua")) {
            // Redirect project loading request to Lua_Wrapper_service and load new project to megamol graph
            /// GUI graph and GUI state are updated at next synchronization
            this->state.request_load_projet_file = filename;
        }
        MinimalPopUp::PopUp("Failed to Load Project", popup_failed, "See console log output for more information.", "",
            confirmed, "Cancel", aborted);
    }
    this->state.open_popup_load = false;
    this->hotkeys[GUIWindows::GuiHotkeyIndex::LOAD_PROJECT].is_pressed = false;

    // File name for screenshot pop-up
    if (this->file_browser.PopUp(this->state.screenshot_filepath, FileBrowserWidget::FileBrowserFlag::SAVE,
            "Select Filename for Screenshot", this->state.open_popup_screenshot, "png")) {
        this->state.screenshot_filepath_id = 0;
    }
    this->state.open_popup_screenshot = false;
}


void megamol::gui::GUIWindows::window_sizing_and_positioning(
    WindowCollection::WindowConfiguration& wc, bool& out_collapsing_changed) {

    ImGuiIO& io = ImGui::GetIO();
    ImVec2 viewport = io.DisplaySize;
    out_collapsing_changed = false;
    float y_offset = (this->state.menu_visible) ? (ImGui::GetFrameHeight()) : (0.0f);
    ImVec2 window_viewport = ImVec2(viewport.x, viewport.y - y_offset);
    bool window_maximized = (wc.win_size == window_viewport);
    bool toggle_window_size = false; // (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0));

    // Context Menu
    if (ImGui::BeginPopupContextItem()) {
        if (ImGui::MenuItem(((window_maximized) ? ("Minimize") : ("Maximize")))) {
            toggle_window_size = true;
        }
        if (ImGui::MenuItem(((!wc.win_collapsed) ? ("Collapse") : ("Expand")), "Double Left Click")) {
            wc.win_collapsed = !wc.win_collapsed;
            out_collapsing_changed = true;
        }

        if (ImGui::MenuItem("Full Width", nullptr)) {
            wc.win_size.x = viewport.x;
            wc.buf_set_pos_size = true;
        }
        ImGui::Separator();

/// DOCKING
#if (defined(IMGUI_HAS_VIEWPORT) && defined(IMGUI_HAS_DOCK))
        ImGui::MenuItem("Docking", "Shift + Left-Drag", false, false);
        ImGui::Separator();
#endif
        ImGui::MenuItem("Snap", nullptr, false, false);

        if (ImGui::ArrowButton("dock_left", ImGuiDir_Left)) {
            wc.win_position.x = 0.0f;
            wc.buf_set_pos_size = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::ArrowButton("dock_up", ImGuiDir_Up)) {
            wc.win_position.y = 0.0f;
            wc.buf_set_pos_size = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::ArrowButton("dock_down", ImGuiDir_Down)) {
            wc.win_position.y = viewport.y - wc.win_size.y;
            wc.buf_set_pos_size = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::ArrowButton("dock_right", ImGuiDir_Right)) {
            wc.win_position.x = viewport.x - wc.win_size.x;
            wc.buf_set_pos_size = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::Separator();

        if (ImGui::MenuItem("Close", nullptr)) {
            wc.win_show = false;
        }
        ImGui::EndPopup();
    }

    // Toggle window size
    if (toggle_window_size) {
        if (window_maximized) {
            // Window is maximized
            wc.win_size = wc.win_reset_size;
            wc.win_position = wc.win_reset_position;
            wc.buf_set_pos_size = true;
        } else {
            // Window is minimized
            ImVec2 window_viewport = ImVec2(viewport.x, viewport.y - y_offset);
            wc.win_reset_size = wc.win_size;
            wc.win_reset_position = wc.win_position;
            wc.win_size = window_viewport;
            wc.win_position = ImVec2(0.0f, y_offset);
            wc.buf_set_pos_size = true;
        }
    }

    // Apply window position and size
    if (wc.buf_set_pos_size || (this->state.menu_visible && ImGui::IsMouseReleased(0) &&
                                   ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows))) {
        this->window_collection.SetWindowSizePosition(wc, this->state.menu_visible);
        wc.buf_set_pos_size = false;
    }
}


bool GUIWindows::considerModule(const std::string& modname, std::vector<std::string>& modules_list) {
    bool retval = false;
    // Empty module list means that all modules should be considered.
    if (modules_list.empty()) {
        retval = true;
    } else {
        retval = (std::find(modules_list.begin(), modules_list.end(), modname) != modules_list.end());
    }
    return retval;
}


void GUIWindows::checkMultipleHotkeyAssignement(void) {
    if (this->state.hotkeys_check_once) {

        std::list<core::view::KeyCode> hotkeylist;
        hotkeylist.clear();

        // Fill with camera hotkeys for which no button parameters exist
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_W));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_A));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_S));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_D));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_C));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_V));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_Q));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_E));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_UP));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_DOWN));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_LEFT));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_RIGHT));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_W, core::view::Modifier::ALT));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_A, core::view::Modifier::ALT));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_S, core::view::Modifier::ALT));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_D, core::view::Modifier::ALT));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_C, core::view::Modifier::ALT));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_V, core::view::Modifier::ALT));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_Q, core::view::Modifier::ALT));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_E, core::view::Modifier::ALT));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_UP, core::view::Modifier::ALT));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_DOWN, core::view::Modifier::ALT));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_LEFT, core::view::Modifier::ALT));
        hotkeylist.emplace_back(core::view::KeyCode(core::view::Key::KEY_RIGHT, core::view::Modifier::ALT));

        // Add hotkeys of gui
        for (auto& h : this->hotkeys) {
            hotkeylist.emplace_back(h.keycode);
        }

        // Add hotkeys of configurator
        for (auto& h : this->configurator.GetHotkeys()) {
            hotkeylist.emplace_back(h.keycode);
        }

        if (auto graph_ptr = this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid)) {
            for (auto& module_ptr : graph_ptr->Modules()) {
                for (auto& param : module_ptr->Parameters()) {

                    if (param.Type() == Param_t::BUTTON) {
                        auto keyCode = param.GetStorage<megamol::core::view::KeyCode>();
                        // Ignore not set hotekey
                        if (keyCode.key == core::view::Key::KEY_UNKNOWN) {
                            break;
                        }
                        // Check in hotkey map
                        bool found = false;
                        for (auto& kc : hotkeylist) {
                            if ((kc.key == keyCode.key) && (kc.mods.equals(keyCode.mods))) {
                                found = true;
                            }
                        }
                        if (!found) {
                            hotkeylist.emplace_back(keyCode);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                                "[GUI] The hotkey [%s] of the parameter \"%s\" has already been assigned. "
                                ">>> If this hotkey is pressed, there will be no effect on this parameter!",
                                keyCode.ToString().c_str(), param.FullName().c_str());
                        }
                    }
                }
            }
        }

        this->state.hotkeys_check_once = false;
    }
}


bool megamol::gui::GUIWindows::isHotkeyPressed(megamol::core::view::KeyCode keycode) {
    ImGuiIO& io = ImGui::GetIO();

    return (ImGui::IsKeyDown(static_cast<int>(keycode.key))) &&
           (keycode.mods.test(core::view::Modifier::ALT) == io.KeyAlt) &&
           (keycode.mods.test(core::view::Modifier::CTRL) == io.KeyCtrl) &&
           (keycode.mods.test(core::view::Modifier::SHIFT) == io.KeyShift);
}


void megamol::gui::GUIWindows::triggerCoreInstanceShutdown(void) {

    if ((this->core_instance != nullptr) && core_instance->IsmmconsoleFrontendCompatible()) { /// mmconsole
#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Shutdown MegaMol instance.");
#endif // GUI_VERBOSE
        this->core_instance->Shutdown();
    }
}


std::string megamol::gui::GUIWindows::project_to_lua_string(void) {

    std::string gui_state;
    if (this->state_to_string(gui_state)) {
        std::string state = std::string(GUI_START_TAG_SET_GUI_VISIBILITY) +
                            ((this->state.gui_visible) ? ("true") : ("false")) +
                            std::string(GUI_END_TAG_SET_GUI_VISIBILITY) + "\n";

        state += std::string(GUI_START_TAG_SET_GUI_SCALE) + std::to_string(megamol::gui::gui_scaling.Get()) +
                 std::string(GUI_END_TAG_SET_GUI_SCALE) + "\n";

        state += std::string(GUI_START_TAG_SET_GUI_STATE) + gui_state + std::string(GUI_END_TAG_SET_GUI_STATE) + "\n";

        return state;
    }
    return std::string();
}


bool megamol::gui::GUIWindows::state_from_string(const std::string& state) {

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
                auto gui_state = header_item.value();
                megamol::core::utility::get_json_value<bool>(gui_state, {"menu_visible"}, &this->state.menu_visible);
                int style = 0;
                megamol::core::utility::get_json_value<int>(gui_state, {"style"}, &style);
                this->state.style = static_cast<GUIWindows::Styles>(style);
                this->state.style_changed = true;
                megamol::core::utility::get_json_value<std::string>(
                    gui_state, {"font_file_name"}, &this->state.font_file_name);
                megamol::core::utility::get_json_value<int>(gui_state, {"font_size"}, &this->state.font_size);
                this->state.font_apply = true;
                float new_gui_scale = 1.0f;
            }
        }

        // Read window configurations
        this->window_collection.StateFromJSON(state_json);

        // Read configurator and graph state
        this->configurator.StateFromJSON(state_json);

        // Read GUI state of parameters (groups) of running graph
        if (auto graph_ptr = this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid)) {
            for (auto& module_ptr : graph_ptr->Modules()) {
                std::string module_full_name = module_ptr->FullName();
                // Parameter Groups
                module_ptr->GUIParameterGroups().StateFromJSON(state_json, module_full_name);
                // Parameters
                for (auto& param : module_ptr->Parameters()) {
                    std::string param_full_name = module_full_name + "::" + param.FullName();
                    param.StateFromJSON(state_json, param_full_name);
                    param.ForceSetGUIStateDirty();
                }
            }
        }

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


bool megamol::gui::GUIWindows::state_to_string(std::string& out_state) {

    try {
        out_state.clear();
        nlohmann::json json_state;

        // Write GUI state
        json_state[GUI_JSON_TAG_GUI]["menu_visible"] = this->state.menu_visible;
        json_state[GUI_JSON_TAG_GUI]["style"] = static_cast<int>(this->state.style);
        GUIUtils::Utf8Encode(this->state.font_file_name);
        json_state[GUI_JSON_TAG_GUI]["font_file_name"] = this->state.font_file_name;
        GUIUtils::Utf8Decode(this->state.font_file_name);
        json_state[GUI_JSON_TAG_GUI]["font_size"] = this->state.font_size;

        // Write window configuration
        this->window_collection.StateToJSON(json_state);

        // Write the configurator and graph state
        this->configurator.StateToJSON(json_state);

        // Write GUI state of parameters (groups) of running graph
        if (auto graph_ptr = this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid)) {
            for (auto& module_ptr : graph_ptr->Modules()) {
                std::string module_full_name = module_ptr->FullName();
                // Parameter Groups
                module_ptr->GUIParameterGroups().StateToJSON(json_state, module_full_name);
                // Parameters
                for (auto& param : module_ptr->Parameters()) {
                    std::string param_full_name = module_full_name + "::" + param.FullName();
                    param.StateToJSON(json_state, param_full_name);
                }
            }
        }

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


void megamol::gui::GUIWindows::init_state(void) {

    this->state.gui_visible = true;
    this->state.gui_visible_post = true;
    this->state.gui_visible_buffer.clear();
    this->state.gui_hide_next_frame = 0;
    this->state.style = GUIWindows::Styles::DarkColors;
    this->state.rescale_windows = false;
    this->state.style_changed = true;
    this->state.new_gui_state = "";
    this->state.project_script_paths.clear();
    this->state.graph_uid = GUI_INVALID_ID;
    this->state.font_utf8_ranges.clear();
    this->state.load_fonts = false;
    this->state.win_delete = "";
    this->state.last_instance_time = 0.0;
    this->state.open_popup_about = false;
    this->state.open_popup_save = false;
    this->state.open_popup_load = false;
    this->state.open_popup_screenshot = false;
    this->state.menu_visible = true;
    this->state.graph_fonts_reserved = 0;
    this->state.toggle_graph_entry = false;
    this->state.shutdown_triggered = false;
    this->state.screenshot_triggered = false;
    this->state.screenshot_filepath = "megamol_screenshot.png";
    this->state.screenshot_filepath_id = 0;
    this->state.hotkeys_check_once = true;
    this->state.font_apply = false;
    this->state.font_file_name = "";
    this->state.request_load_projet_file = "";
    this->state.stat_averaged_fps = 0.0;
    this->state.stat_averaged_ms = 0.0;
    this->state.stat_frame_count = 0;
    this->state.font_size = 13;
    this->state.resource_directories.clear();

    this->create_not_existing_png_filepath(this->state.screenshot_filepath);
}


void megamol::gui::GUIWindows::update_frame_statistics(WindowCollection::WindowConfiguration& wc) {

    ImGuiIO& io = ImGui::GetIO();

    wc.buf_current_delay += io.DeltaTime;
    if (wc.fpsms_refresh_rate > 0.0f) {
        if (wc.buf_current_delay >= (1.0f / wc.fpsms_refresh_rate)) {

            auto update_values = [](float current_value, float& max_value, std::vector<float>& values,
                                     size_t actual_buffer_size) {
                auto buffer_size = static_cast<int>(values.size());
                if (buffer_size != actual_buffer_size) {
                    if (buffer_size > actual_buffer_size) {
                        values.erase(values.begin(), values.begin() + (buffer_size - actual_buffer_size));

                    } else if (buffer_size < actual_buffer_size) {
                        values.insert(values.begin(), (actual_buffer_size - buffer_size), 0.0f);
                    }
                }
                if (buffer_size > 0) {
                    values.erase(values.begin());
                    values.emplace_back(static_cast<float>(current_value));
                    float new_max_value = 0.0f;
                    for (auto& v : values) {
                        new_max_value = std::max(v, new_max_value);
                    }
                    max_value = new_max_value;
                }
            };

            update_values(
                ((this->state.stat_averaged_fps == 0.0) ? (1.0f / io.DeltaTime) : (this->state.stat_averaged_fps)),
                wc.buf_fps_max, wc.buf_fps_values, wc.fpsms_buffer_size);

            update_values(
                ((this->state.stat_averaged_ms == 0.0) ? (io.DeltaTime * 1000.0f) : (this->state.stat_averaged_ms)),
                wc.buf_ms_max, wc.buf_ms_values, wc.fpsms_buffer_size);

            wc.buf_current_delay = 0.0f;
        }
    }
}


bool megamol::gui::GUIWindows::create_not_existing_png_filepath(std::string& inout_filepath) {

    // Check for existing file
    bool created_filepath = false;
    if (!inout_filepath.empty()) {
        while (megamol::core::utility::FileUtils::FileExists<std::string>(inout_filepath)) {
            // Create new filename with iterating suffix
            std::string filename = megamol::core::utility::FileUtils::GetFilenameStem<std::string>(inout_filepath);
            std::string id_separator = "_";
            bool new_separator = false;
            auto separator_index = filename.find_last_of(id_separator);
            if (separator_index != std::string::npos) {
                auto last_id_str = filename.substr(separator_index + 1);
                try {
                    this->state.screenshot_filepath_id = std::stoi(last_id_str);
                } catch (...) { new_separator = true; }
                this->state.screenshot_filepath_id++;
                if (new_separator) {
                    this->state.screenshot_filepath =
                        filename + id_separator + std::to_string(this->state.screenshot_filepath_id) + ".png";
                } else {
                    inout_filepath = filename.substr(0, separator_index + 1) +
                                     std::to_string(this->state.screenshot_filepath_id) + ".png";
                }
            } else {
                inout_filepath = filename + id_separator + std::to_string(this->state.screenshot_filepath_id) + ".png";
            }
        }
        created_filepath = true;
    }
    return created_filepath;
}

void megamol::gui::GUIWindows::ShowTextures() {
    auto render_image = [&](std::string const& name, unsigned int gl_texture, unsigned int width, unsigned int height) {
        ImGui::Begin((name + " Rendering Result").c_str(), nullptr, ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::Image((ImTextureID) gl_texture, ImVec2(width, height), ImVec2(0, 1), ImVec2(1, 0));
        ImGui::End();
    };

    #define val(X) std::get<X>(image)

    for (auto& image : m_textures_test)
        render_image(val(0), val(1), val(2), val(3));

    #undef val
}

void megamol::gui::GUIWindows::ShowHeadnodeRemoteControl() {
    if (!m_headnode_remote_control)
        return;

    static bool headnode_running = false;
    static std::string lua_command = "";
    static std::string param_send_modules = "all";
    static bool keep_sending_params = false;


    auto command_value = [&](unsigned int cmd, std::string const& value) {
        (*this->m_headnode_remote_control)(cmd, value);
    };
    auto command = [&](unsigned int cmd) {
        command_value(cmd, "");
    };

    auto window_name = "Head Node Remote Control";
    ImGui::Begin(window_name, nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    {
        if (headnode_running) {
            ImGui::Text("(Server running) ");
            ImGui::SameLine();
            // CloseHeadNode - 2
            if (ImGui::Button("Stop Head Node Server")) {
                command(2);
                headnode_running = false;
            }
        } else {
            ImGui::Text("(Server stopped) ");
            ImGui::SameLine();
            //  StartHeadNode - 1
            if (ImGui::Button("Start Head Node Server")) {
                command(1);
                headnode_running = true;
            }
        }

        // ClearGraph - 3
        if (ImGui::Button("Clear Rendernode Graphs")) {
            command(3);
            keep_sending_params = false;
            command(6);
        }
        ImGui::SameLine();
        // SendGraph  - 4
        if (ImGui::Button("Broadcast Local Graph"))
            command(4);

        if (ImGui::RadioButton("Sync Module Params", keep_sending_params)) {
            keep_sending_params = !keep_sending_params;
            // KeepSendingParams - 5
            // DontSendParams    - 6
            keep_sending_params
                ? command(5)
                : command(6);
        }
        ImGui::SameLine();
        if (ImGui::InputText("Sync Modules", &param_send_modules, ImGuiInputTextFlags_EnterReturnsTrue)) {
            command_value(7, param_send_modules);
        }

        if (ImGui::Button("Send Lua Command")) {
            // SendLuaCommand - 8
            command_value(8, lua_command);
        }
        ImGui::SameLine();
        ImGui::InputText("Lua Command", &lua_command);

    }
    ImGui::End();
}

