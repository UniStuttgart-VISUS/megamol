/*
 * GUIWindows.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

/**
 * USED HOTKEYS:
 *
 * - Trigger Screenshot:        F2
 * - Toggle Main View:          F3
 * - Reset Windows Positions:   F4
 * - Show/hide Windows:         F6-F11
 * - Show/hide Menu:            F12
 * - Search Parameter:          Ctrl  + p
 * - Save Running Project:      Ctrl  + s
 * - Quit Program:              Alt   + F4
 */

#include "stdafx.h"
#include "GUIWindows.h"


using namespace megamol;
using namespace megamol::gui;


GUIWindows::GUIWindows(void)
        : core_instance(nullptr)
        , hotkeys()
        , context(nullptr)
        , initialized_api(GUIImGuiAPI::NONE)
        , window_collection()
        , configurator()
        , console()
        , state()
        , file_browser()
        , search_widget()
        , tf_editor_ptr(nullptr)
        , tooltip()
        , picking_buffer()
        , triangle_widget() {

    this->hotkeys[GUIWindows::GuiHotkeyIndex::TRIGGER_SCREENSHOT] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F2, core::view::Modifier::NONE), false};
    this->hotkeys[GUIWindows::GuiHotkeyIndex::TOGGLE_MAIN_VIEWS] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F3, core::view::Modifier::NONE), false};
    this->hotkeys[GUIWindows::GuiHotkeyIndex::EXIT_PROGRAM] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F4, core::view::Modifier::ALT), false};
    this->hotkeys[GUIWindows::GuiHotkeyIndex::RESET_WINDOWS_POS] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F4, core::view::Modifier::NONE), false};
    this->hotkeys[GUIWindows::GuiHotkeyIndex::MENU] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F12, core::view::Modifier::NONE), false};
    this->hotkeys[GUIWindows::GuiHotkeyIndex::PARAMETER_SEARCH] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_P, core::view::Modifier::CTRL), false};
    this->hotkeys[GUIWindows::GuiHotkeyIndex::SAVE_PROJECT] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_S, core::view::Modifier::CTRL), false};
    this->hotkeys[GUIWindows::GuiHotkeyIndex::LOAD_PROJECT] = {
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_L, core::view::Modifier::CTRL), false};

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


bool GUIWindows::CreateContext_GL(megamol::core::CoreInstance* instance) {

    if (instance == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Pointer to core instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->core_instance = instance;

    return this->createContext(GUIImGuiAPI::OPEN_GL);
}


bool GUIWindows::PreDraw(glm::vec2 framebuffer_size, glm::vec2 window_size, double instance_time) {

    /// [DEPRECATED USAGE] ///
    // Disable GUI drawing if GUIView module is chained
    if (this->state.gui_enabled && ImGui::GetCurrentContext()->WithinFrameScope) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Chaining GUIVIew modules is not supported. GUI is disabled. [%s, %s, line %d]\n", __FILE__,
            __FUNCTION__, __LINE__);
        this->state.gui_enabled = false;
    }

    // Required to prevent change in gui drawing between pre and post draw
    this->state.enable_gui_post = this->state.gui_enabled;

    // Early exit when pre step should be omitted
    if (!this->state.gui_enabled) {
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

    // Propagate ImGui context to core instance
    if (this->core_instance != nullptr) {
        this->core_instance->SetCurrentImGuiContext(this->context);
    }

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

    // Set stuff for next frame --------------------------------------------

    // IO
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

    // Process hotkeys
    this->checkMultipleHotkeyAssignement();
    if (this->hotkeys[GUIWindows::GuiHotkeyIndex::EXIT_PROGRAM].is_pressed) {
        this->triggerCoreInstanceShutdown();
        this->state.shutdown_triggered = true;
        return true;
    }
    if (this->hotkeys[GUIWindows::GuiHotkeyIndex::MENU].is_pressed) {
        this->state.menu_visible = !this->state.menu_visible;
        this->hotkeys[GUIWindows::GuiHotkeyIndex::MENU].is_pressed = false;
    }
    if (this->hotkeys[GUIWindows::GuiHotkeyIndex::TRIGGER_SCREENSHOT].is_pressed) {
        this->state.screenshot_triggered = true;
        this->hotkeys[GUIWindows::GuiHotkeyIndex::TRIGGER_SCREENSHOT].is_pressed = false;
    }
    if (this->state.toggle_main_view || this->hotkeys[GUIWindows::GuiHotkeyIndex::TOGGLE_MAIN_VIEWS].is_pressed) {
        GraphPtr_t graph_ptr;
        if (this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid, graph_ptr)) {
            megamol::gui::ModulePtrVector_t::const_iterator module_mainview_iter = graph_ptr->GetModules().begin();
            // Search for first main view and set next view to main view (= graph entry point)
            for (auto module_iter = graph_ptr->GetModules().begin(); module_iter != graph_ptr->GetModules().end();
                 module_iter++) {
                if ((*module_iter)->is_view && (*module_iter)->IsMainView()) {
                    // Remove all main views
                    (*module_iter)->main_view_name.clear();
                    Graph::QueueData queue_data;
                    queue_data.name_id = (*module_iter)->FullName();
                    graph_ptr->PushSyncQueue(Graph::QueueAction::REMOVE_MAIN_VIEW, queue_data);
                    // Save index of last found main view
                    if (module_iter != graph_ptr->GetModules().end()) {
                        module_mainview_iter = module_iter + 1;
                    }
                }
            }
            if ((module_mainview_iter == graph_ptr->GetModules().begin()) ||
                (module_mainview_iter != graph_ptr->GetModules().end())) {
                // Search for next main view
                for (auto module_iter = module_mainview_iter; module_iter != graph_ptr->GetModules().end();
                     module_iter++) {
                    if ((*module_iter)->is_view) {
                        (*module_iter)->main_view_name = graph_ptr->GenerateUniqueMainViewName();
                        Graph::QueueData queue_data;
                        queue_data.name_id = (*module_iter)->FullName();
                        graph_ptr->PushSyncQueue(Graph::QueueAction::CREATE_MAIN_VIEW, queue_data);
                        break;
                    }
                }
            }
        }
        this->state.toggle_main_view = false;
        this->hotkeys[GUIWindows::GuiHotkeyIndex::TOGGLE_MAIN_VIEWS].is_pressed = false;
    }

    // Auto-save state
    this->state.win_save_delay += io.DeltaTime;
    if (this->state.autosave_gui_state && this->state.win_save_state && (this->state.win_save_delay > 1.0f)) {
        // Delayed saving after triggering saving state (in seconds).
        GraphPtr_t graph_ptr;
        if (this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid, graph_ptr)) {
            std::string filename = graph_ptr->GetFilename();
            this->configurator.GetGraphCollection().SaveProjectToFile(
                this->state.graph_uid, filename, this->dump_state_to_file(filename));
        }
        this->state.win_save_state = false;
    }

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

    // Loading new font (set in FONT window)
    if (!this->state.font_file.empty()) {
        ImFontConfig config;
        config.OversampleH = 4;
        config.OversampleV = 4;
        config.GlyphRanges = this->state.font_utf8_ranges.data();

        GUIUtils::Utf8Encode(this->state.font_file);
        io.Fonts->AddFontFromFileTTF(this->state.font_file.c_str(), this->state.font_size, &config);
        ImGui_ImplOpenGL3_CreateFontsTexture();
        // Load last added font
        io.FontDefault = io.Fonts->Fonts[(io.Fonts->Fonts.Size - 1)];
        this->state.font_file.clear();
    }

    // Loading new font from state (set in loaded FONT window configuration)
    if (this->state.font_index >= this->state.graph_fonts_reserved) {
        if (this->state.font_index < static_cast<unsigned int>(io.Fonts->Fonts.Size)) {
            io.FontDefault = io.Fonts->Fonts[this->state.font_index];
        }
        this->state.font_index = GUI_INVALID_ID;
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
    if (!this->state.enable_gui_post) {
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

    ////////// DRAW ///////////////////////////////////////////////////////////

    // Main Menu ---------------------------------------------------------------
    if (this->state.menu_visible) {

        if (ImGui::BeginMainMenuBar()) {
            this->drawMenu();
            ImGui::EndMainMenuBar();
        }
    }

    // Global Docking Space ---------------------------------------------------
    /// DOCKING
#if (defined(IMGUI_HAS_VIEWPORT) && defined(IMGUI_HAS_DOCK))
    ImGuiStyle& style = ImGui::GetStyle();
    auto child_bg = style.Colors[ImGuiCol_ChildBg];
    style.Colors[ImGuiCol_ChildBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);
    style.Colors[ImGuiCol_ChildBg] = child_bg;
#endif

    // Draw Windows ------------------------------------------------------------
    const auto func = [&, this](WindowCollection::WindowConfiguration& wc) {
        // Loading changed window state of font (even if window is not shown)
        if ((wc.win_callback == WindowCollection::DrawCallbacks::FONT) && wc.buf_font_reset) {
            if (!wc.font_name.empty()) {
                this->state.font_index = GUI_INVALID_ID;
                for (unsigned int n = this->state.graph_fonts_reserved;
                     n < static_cast<unsigned int>(io.Fonts->Fonts.Size); n++) {
                    std::string font_name = std::string(io.Fonts->Fonts[n]->GetDebugName());
                    GUIUtils::Utf8Decode(font_name);
                    if (font_name == wc.font_name) {
                        this->state.font_index = n;
                    }
                }
                if (this->state.font_index == GUI_INVALID_ID) {
                    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                        "[GUI] Could not find font '%s' for loaded state. [%s, %s, line %d]\n", wc.font_name.c_str(),
                        __FILE__, __FUNCTION__, __LINE__);
                }
            }
            wc.buf_font_reset = false;
        }

        // Loading changed window state of transfer function editor (even if window is not shown)
        if ((wc.win_callback == WindowCollection::DrawCallbacks::TRANSFER_FUNCTION) && wc.buf_tfe_reset) {
            this->tf_editor_ptr->SetMinimized(wc.tfe_view_minimized);
            this->tf_editor_ptr->SetVertical(wc.tfe_view_vertical);

            if (!wc.tfe_active_param.empty()) {
                GraphPtr_t graph_ptr;
                if (this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid, graph_ptr)) {
                    for (auto& module_ptr : graph_ptr->GetModules()) {
                        std::string module_full_name = module_ptr->FullName();
                        for (auto& param : module_ptr->parameters) {
                            std::string param_full_name = module_full_name + "::" + param.full_name;
                            if ((wc.tfe_active_param == param_full_name) && (param.type == Param_t::TRANSFERFUNCTION)) {
                                this->tf_editor_ptr->SetConnectedParameter(&param, param_full_name);
                                this->tf_editor_ptr->SetTransferFunction(std::get<std::string>(param.GetValue()), true);
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

            wc.win_soft_reset |= this->hotkeys[GUIWindows::GuiHotkeyIndex::RESET_WINDOWS_POS].is_pressed;

            // Begin Window
            auto window_title = wc.win_name + "     " + wc.win_hotkey.ToString();
            if (!ImGui::Begin(window_title.c_str(), &wc.win_show, wc.win_flags)) {
                wc.win_collapsed = ImGui::IsWindowCollapsed();
                ImGui::End(); // early ending
                return;
            }

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
            wc.win_position = ImGui::GetWindowPos();
            wc.win_size = ImGui::GetWindowSize();
            if (!collapsing_changed) {
                wc.win_collapsed = ImGui::IsWindowCollapsed();
            }

            ImGui::End();
        }
    };
    this->window_collection.EnumWindows(func);

    // Draw global parameter widgets -------------------------------------------

    /// DEBUG TEST OpenGL Picking
    // auto viewport_dim = glm::vec2(io.DisplaySize.x, io.DisplaySize.y);
    // this->picking_buffer.EnableInteraction(viewport_dim);

    GraphPtr_t graph_ptr;
    if (this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid, graph_ptr)) {
        for (auto& module_ptr : graph_ptr->GetModules()) {

            /// DEBUG TEST OpenGL Picking
            /// TODO Pass picked UID to parameters

            module_ptr->present.param_groups.PresentGUI(module_ptr->parameters, module_ptr->FullName(), "",
                vislib::math::Ternary(vislib::math::Ternary::TRI_UNKNOWN), false,
                ParameterPresentation::WidgetScope::GLOBAL, this->tf_editor_ptr, nullptr);
        }
    }

    /// DEBUG TEST OpenGL Picking
    // unsigned int id = 5;
    // this->picking_buffer.AddInteractionObject(id, this->triangle_widget.GetInteractions(id));
    // this->triangle_widget.Draw(
    //    id, glm::vec2(0.0f, 200.0f), viewport_dim, this->picking_buffer.GetPendingManipulations());
    // this->picking_buffer.DisableInteraction();

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
    GraphPtr_t graph_ptr;
    if (this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid, graph_ptr)) {
        for (auto& module_ptr : graph_ptr->GetModules()) {
            // Break loop after first occurrence of parameter hotkey
            if (hotkeyPressed)
                break;
            if (check_all_modules || this->considerModule(module_ptr->FullName(), modules_list)) {
                for (auto& param : module_ptr->parameters) {
                    if (param.type == Param_t::BUTTON) {
                        auto keyCode = param.GetStorage<megamol::core::view::KeyCode>();
                        if (this->isHotkeyPressed(keyCode)) {
                            param.ForceSetValueDirty();
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

    // Trigger saving state when mouse hovered any window and on button mouse release event
    if (!io.MouseDown[buttonIndex] && ImGui::IsWindowHovered(hoverFlags)) {
        this->state.win_save_state = true;
        this->state.win_save_delay = 0.0f;
    }

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


bool megamol::gui::GUIWindows::ConsumeTriggeredScreenshot(void) {

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


bool megamol::gui::GUIWindows::SynchronizeGraphs(megamol::core::MegaMolGraph* megamol_graph) {

    // Disable synchronizing graphs when pre step is omitted
    if (!this->state.gui_enabled) {
        return true;
    }

    // 1) Load all known calls and modules from core instance ONCE ---------------------------
    if (!this->configurator.GetGraphCollection().LoadCallStock(core_instance)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Failed to load call stock once. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (!this->configurator.GetGraphCollection().LoadModuleStock(core_instance)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Failed to load module stock once. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    bool synced = false;
    bool sync_success = false;
    GraphPtr_t graph_ptr;
    bool found_graph = this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid, graph_ptr);
    // 2a) Either synchronize GUI Graph -> Core Graph ... ---------------------
    if (!synced && found_graph) {
        bool graph_sync_success = true;

        Graph::QueueAction action;
        Graph::QueueData data;
        while (graph_ptr->PopSyncQueue(action, data)) {
            synced = true;
            switch (action) {
            case (Graph::QueueAction::ADD_MODULE): {
                if (megamol_graph != nullptr) {
                    graph_sync_success &= megamol_graph->CreateModule(data.class_name, data.name_id);
                } else if (this->core_instance != nullptr) {
                    graph_sync_success &= this->core_instance->RequestModuleInstantiation(
                        vislib::StringA(data.class_name.c_str()), vislib::StringA(data.name_id.c_str()));
                }
            } break;
            case (Graph::QueueAction::RENAME_MODULE): {
                if (megamol_graph != nullptr) {
                    bool rename_success = megamol_graph->RenameModule(data.name_id, data.rename_id);
                    graph_sync_success &= rename_success;
                } else if (this->core_instance != nullptr) {
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
                } else if (this->core_instance != nullptr) {
                    graph_sync_success &=
                        this->core_instance->RequestCallInstantiation(vislib::StringA(data.class_name.c_str()),
                            vislib::StringA(data.caller.c_str()), vislib::StringA(data.callee.c_str()));
                }
            } break;
            case (Graph::QueueAction::DELETE_CALL): {
                if (megamol_graph != nullptr) {
                    graph_sync_success &= megamol_graph->DeleteCall(data.caller, data.callee);
                } else if (this->core_instance != nullptr) {
                    graph_sync_success &= this->core_instance->RequestCallDeletion(
                        vislib::StringA(data.caller.c_str()), vislib::StringA(data.callee.c_str()));
                }
            } break;
            case (Graph::QueueAction::CREATE_MAIN_VIEW): {
                if (megamol_graph != nullptr) {
                    megamol_graph->SetGraphEntryPoint(data.name_id,
                        megamol::core::view::get_gl_view_runtime_resources_requests(),
                        megamol::core::view::view_rendering_execution, megamol::core::view::view_init_rendering_state);
                } else if (this->core_instance != nullptr) {
                    /* XXX Currently not supported by core graph
                     */
                }
            } break;
            case (Graph::QueueAction::REMOVE_MAIN_VIEW): {
                if (megamol_graph != nullptr) {
                    megamol_graph->RemoveGraphEntryPoint(data.name_id);
                } else if (this->core_instance != nullptr) {
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
        bool graph_sync_success =
            this->configurator.GetGraphCollection().LoadUpdateProjectFromCore(this->state.graph_uid,
                ((megamol_graph == nullptr) ? (this->core_instance) : (nullptr)), megamol_graph, true);
        if (!graph_sync_success) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Failed to synchronize core graph with gui graph. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
                __LINE__);
        }

        GraphPtr_t graph_ptr;
        if (graph_sync_success && this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid, graph_ptr)) {
            std::string script_filename = this->state.last_script_filename;
            if (graph_ptr->GetFilename().empty()) {
                if (this->core_instance != nullptr) {
                    // Set project filename from lua state of core instance
                    script_filename = this->core_instance->GetLuaState()->GetScriptPath();
                }
            }
            // Always check for changed script path when project file is dropped
            if (!this->state.project_script_paths.empty()) {
                script_filename = this->state.project_script_paths.front();
            }
            // Load GUI state from project file when project file changed
            if (script_filename != this->state.last_script_filename) {
                graph_ptr->SetFilename(script_filename);
                this->load_state_from_file(graph_ptr->GetFilename());
                this->state.last_script_filename = script_filename;
            }
        }
        sync_success &= graph_sync_success;
    }

    // 3) Synchronize parameter values -------------------------------------------
    if (found_graph) {
        bool param_sync_success = true;
        for (auto& module_ptr : graph_ptr->GetModules()) {
            for (auto& param : module_ptr->parameters) {

                // Try to connect gui parameters to newly created parameters of core modules
                if (param.core_param_ptr.IsNull()) {
                    auto module_name = module_ptr->FullName();
                    megamol::core::Module* core_module_ptr = nullptr;
                    if (megamol_graph != nullptr) {
                        core_module_ptr = megamol_graph->FindModule(module_name).get();
                    } else if (this->core_instance != nullptr) {
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
                                for (auto& parameter : module_ptr->parameters) {
                                    if (parameter.full_name == param_full_name) {
                                        megamol::gui::Parameter::ReadNewCoreParameterToExistingParameter(
                                            (*param_slot), parameter, true, false, true);
                                    }
                                }
                            }
                        }
                    }
#ifdef GUI_VERBOSE
                    if (param.core_param_ptr.IsNull()) {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] Unable to connect core parameter to gui parameter. [%s, %s, line %d]\n", __FILE__,
                            __FUNCTION__, __LINE__);
                    }
#endif // GUI_VERBOSE
                }

                if (!param.core_param_ptr.IsNull()) {
                    // Write changed gui state to core parameter
                    if (param.present.IsGUIStateDirty()) {
                        param_sync_success &=
                            megamol::gui::Parameter::WriteCoreParameterGUIState(param, param.core_param_ptr);
                        param.present.ResetGUIStateDirty();
                    }
                    // Write changed parameter value to core parameter
                    if (param.IsValueDirty()) {
                        param_sync_success &=
                            megamol::gui::Parameter::WriteCoreParameterValue(param, param.core_param_ptr);
                        param.ResetValueDirty();
                    }
                    // Read current parameter value and GUI state fro core parameter
                    param_sync_success &= megamol::gui::Parameter::ReadCoreParameterToParameter(
                        param.core_param_ptr, param, false, false);
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


bool GUIWindows::createContext(GUIImGuiAPI imgui_api) {

    if (this->initialized_api != GUIImGuiAPI::NONE) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] ImGui context has alreday been created. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return true;
    }

    // Check for existing context and share FontAtlas with new context (required by ImGui).
    bool other_context_exists = (ImGui::GetCurrentContext() != nullptr);
    ImFontAtlas* font_atlas = nullptr;
    ImFont* default_font = nullptr;

    /// [DEPRECATED USAGE] ///
    if (other_context_exists) {
        ImGuiIO& current_io = ImGui::GetIO();
        font_atlas = current_io.Fonts;
        default_font = current_io.FontDefault;
        ImGui::GetCurrentContext()->FontAtlasOwnedByContext = false;
        // Init API only once (is it same as the already initialized one?)
        this->initialized_api = imgui_api;
    }

    // Create ImGui context ---------------------------------------------------
    IMGUI_CHECKVERSION();
    this->context = ImGui::CreateContext(font_atlas);
    if (this->context == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unable to create ImGui context. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    megamol::gui::imgui_context_count++;
    ImGui::SetCurrentContext(this->context);

    // Create ImGui API
    if (!other_context_exists) {
        switch (imgui_api) {
        case (GUIImGuiAPI::OPEN_GL): {
            // Init OpenGL for ImGui
            const char* glsl_version = "#version 130"; /// "#version 150" or nullptr
            if (ImGui_ImplOpenGL3_Init(glsl_version)) {
                this->initialized_api = GUIImGuiAPI::OPEN_GL;
                megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Created ImGui context for Open GL.");
            }
        } break;
        default: {
            this->destroyContext();
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] ImGui API is not supported yet. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        } break;
        }
    }

    // Register window callbacks in window collection -------------------------
    this->window_collection.RegisterDrawWindowCallback(WindowCollection::DrawCallbacks::MAIN_PARAMETERS,
        [&, this](WindowCollection::WindowConfiguration& wc) { this->drawParamWindowCallback(wc); });
    this->window_collection.RegisterDrawWindowCallback(WindowCollection::DrawCallbacks::PARAMETERS,
        [&, this](WindowCollection::WindowConfiguration& wc) { this->drawParamWindowCallback(wc); });
    this->window_collection.RegisterDrawWindowCallback(WindowCollection::DrawCallbacks::PERFORMANCE,
        [&, this](WindowCollection::WindowConfiguration& wc) { this->drawFpsWindowCallback(wc); });
    this->window_collection.RegisterDrawWindowCallback(WindowCollection::DrawCallbacks::FONT,
        [&, this](WindowCollection::WindowConfiguration& wc) { this->drawFontWindowCallback(wc); });
    this->window_collection.RegisterDrawWindowCallback(WindowCollection::DrawCallbacks::TRANSFER_FUNCTION,
        [&, this](WindowCollection::WindowConfiguration& wc) { this->drawTransferFunctionWindowCallback(wc); });
    this->window_collection.RegisterDrawWindowCallback(WindowCollection::DrawCallbacks::CONFIGURATOR,
        [&, this](WindowCollection::WindowConfiguration& wc) { this->drawConfiguratorWindowCallback(wc); });
    this->window_collection.RegisterDrawWindowCallback(WindowCollection::DrawCallbacks::LOGCONSOLE,
        [&, this](WindowCollection::WindowConfiguration& wc) { this->console.Draw(wc); });

    // Create window configurations
    WindowCollection::WindowConfiguration buf_win;
    buf_win.win_store_config = true;
    buf_win.win_reset = true;
    buf_win.win_position = ImVec2(0.0f, 0.0f);
    buf_win.win_reset_position = ImVec2(0.0f, 0.0f);
    buf_win.win_size = ImVec2(400.0f, 600.0f);

    // MAIN Window ------------------------------------------------------------
    buf_win.win_name = "All Parameters";
    buf_win.win_show = true;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F11);
    buf_win.win_flags = ImGuiWindowFlags_NoScrollbar;
    buf_win.win_callback = WindowCollection::DrawCallbacks::MAIN_PARAMETERS;
    buf_win.win_reset_size = buf_win.win_size;
    this->window_collection.AddWindowConfiguration(buf_win);

    // FPS/MS Window ----------------------------------------------------------
    buf_win.win_name = "Performance Metrics";
    buf_win.win_show = false;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F10);
    buf_win.win_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
    buf_win.win_callback = WindowCollection::DrawCallbacks::PERFORMANCE;
    this->window_collection.AddWindowConfiguration(buf_win);

    // FONT Window ------------------------------------------------------------
    buf_win.win_name = "Font Settings";
    buf_win.win_show = false;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F9);
    buf_win.win_flags = ImGuiWindowFlags_AlwaysAutoResize;
    buf_win.win_callback = WindowCollection::DrawCallbacks::FONT;
    this->window_collection.AddWindowConfiguration(buf_win);

    // TRANSFER FUNCTION Window -----------------------------------------------
    buf_win.win_name = "Transfer Function Editor";
    buf_win.win_show = false;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F8);
    buf_win.win_flags = ImGuiWindowFlags_AlwaysAutoResize;
    buf_win.win_callback = WindowCollection::DrawCallbacks::TRANSFER_FUNCTION;
    this->window_collection.AddWindowConfiguration(buf_win);

    // CONFIGURATOR Window -----------------------------------------------
    buf_win.win_name = "Configurator";
    buf_win.win_show = false;
    /// TODO Better initial size for configurator (use current viewport?)
    buf_win.win_size = ImVec2(800.0f, 600.0f);
    buf_win.win_reset_size = buf_win.win_size;
    buf_win.win_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoScrollbar;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F7);
    buf_win.win_callback = WindowCollection::DrawCallbacks::CONFIGURATOR;
    this->window_collection.AddWindowConfiguration(buf_win);

    // LOG CONSOLE Window -----------------------------------------------
    buf_win.win_name = "Log Console";
    buf_win.win_show = false;
    buf_win.win_size = ImVec2(850.0f, 250.0f);
    buf_win.win_reset_size = buf_win.win_size;
    buf_win.win_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_HorizontalScrollbar;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F6);
    buf_win.win_callback = WindowCollection::DrawCallbacks::LOGCONSOLE;
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

        ImGuiIO& io = ImGui::GetIO();
        const float default_font_size = 12.0f;
        ImFontConfig config;
        config.OversampleH = 4;
        config.OversampleV = 4;
        config.GlyphRanges = this->state.font_utf8_ranges.data();
        std::string configurator_font;
        std::string default_font;

        // Add other known fonts
        std::vector<std::string> font_paths;
        if (this->core_instance != nullptr) {
            auto search_paths = this->core_instance->Configuration().ResourceDirectories();
            for (size_t i = 0; i < search_paths.Count(); ++i) {
                std::wstring search_path(search_paths[i].PeekBuffer());
                std::string font_path =
                    FileUtils::SearchFileRecursive<std::wstring, std::string>(search_path, "Roboto-Regular.ttf");
                if (!font_path.empty()) {
                    font_paths.emplace_back(font_path);
                    configurator_font = font_path;
                    default_font = font_path;
                }
                font_path =
                    FileUtils::SearchFileRecursive<std::wstring, std::string>(search_path, "SourceCodePro-Regular.ttf");
                if (!font_path.empty()) {
                    font_paths.emplace_back(font_path);
                }
            }
        }

        // Configurator Graph Font: Add default font at first n indices for exclusive use in configurator graph.
        /// Workaround: Using different font sizes for different graph zooming factors to improve font readability when
        /// zooming.
        const auto graph_font_scalings = this->configurator.GetGraphFontScalings();
        this->state.graph_fonts_reserved = graph_font_scalings.size();
        if (configurator_font.empty()) {
            for (unsigned int i = 0; i < this->state.graph_fonts_reserved; i++) {
                io.Fonts->AddFontDefault(&config);
            }
        } else {
            for (unsigned int i = 0; i < this->state.graph_fonts_reserved; i++) {
                io.Fonts->AddFontFromFileTTF(
                    configurator_font.c_str(), default_font_size * graph_font_scalings[i], &config);
            }
        }

        // Add other fonts for gui.
        io.Fonts->AddFontDefault(&config);
        io.FontDefault = io.Fonts->Fonts[(io.Fonts->Fonts.Size - 1)];
        for (auto& font_path : font_paths) {
            io.Fonts->AddFontFromFileTTF(font_path.c_str(), default_font_size, &config);
            if (default_font == font_path) {
                io.FontDefault = io.Fonts->Fonts[(io.Fonts->Fonts.Size - 1)];
            }
        }
    }

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

    return true;
}


bool GUIWindows::destroyContext(void) {

    if (this->initialized_api != GUIImGuiAPI::NONE) {
        if (this->context != nullptr) {

            /// [DEPRECATED USAGE] ///
            // Shutdown API only if one context is left
            if (megamol::gui::imgui_context_count < 2) {
                ImGui::SetCurrentContext(this->context);

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
            megamol::gui::imgui_context_count--;
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


void GUIWindows::drawTransferFunctionWindowCallback(WindowCollection::WindowConfiguration& wc) {

    this->tf_editor_ptr->Widget(true);
    wc.tfe_active_param = this->tf_editor_ptr->GetConnectedParameterName();
}


void GUIWindows::drawConfiguratorWindowCallback(WindowCollection::WindowConfiguration& wc) {

    this->configurator.Draw(wc, this->core_instance);
}


void GUIWindows::drawParamWindowCallback(WindowCollection::WindowConfiguration& wc) {

    // Mode
    megamol::gui::ParameterPresentation::ParameterExtendedModeButton(wc.param_extended_mode);
    this->tooltip.Marker("Expert mode enables options for additional parameter presentation options.");
    ImGui::SameLine();

    // Options
    ImGuiID overrideState = GUI_INVALID_ID;
    if (ImGui::Button("Expand All")) {
        overrideState = 1; // open
    }
    ImGui::SameLine();

    if (ImGui::Button("Collapse All")) {
        overrideState = 0; // close
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

    const size_t dnd_size = 2048; // Set same max size of all module labels for drag and drop.
    auto current_search_string = this->search_widget.GetSearchString();
    GraphPtr_t graph_ptr;
    // Listing modules and their parameters
    if (this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid, graph_ptr)) {
        for (auto& module_ptr : graph_ptr->GetModules()) {
            std::string module_label = module_ptr->FullName();

            // Check if module should be considered.
            if (!this->considerModule(module_label, wc.param_modules_list)) {
                continue;
            }

            // Determine header state and change color depending on active parameter search
            auto headerId = ImGui::GetID(module_label.c_str());
            auto headerState = overrideState;
            if (headerState == GUI_INVALID_ID) {
                headerState = ImGui::GetStateStorage()->GetInt(headerId, 0); // 0=close 1=open
            }
            auto search_string = current_search_string;
            bool module_searched = true;
            if (!search_string.empty()) {
                headerState = 1;
                module_searched =
                    megamol::gui::StringSearchWidget::FindCaseInsensitiveSubstring(module_label, search_string);
                if (!module_searched) {
                    ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyleColorVec4(ImGuiCol_PopupBg));
                } else {
                    // Show all when module is part of the search
                    search_string.clear();
                }
            }
            ImGui::GetStateStorage()->SetInt(headerId, headerState);

            bool header_open = ImGui::CollapsingHeader(module_label.c_str(), nullptr);

            if (!search_string.empty() && !module_searched) {
                ImGui::PopStyleColor();
            }

            // Module description as hover tooltip
            this->tooltip.ToolTip(module_ptr->description, ImGui::GetID(module_label.c_str()), 0.5f, 5.0f);

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
                    buf_win.win_position = ImVec2(0.0f, ImGui::GetTextLineHeightWithSpacing());
                    buf_win.win_size = ImVec2(400.0f, 600.0f);
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

            if (header_open) {
                // Draw parameters
                bool out_open_external_tf_editor;

                module_ptr->present.param_groups.PresentGUI(module_ptr->parameters, module_label, search_string,
                    vislib::math::Ternary(wc.param_extended_mode), true, ParameterPresentation::WidgetScope::LOCAL,
                    this->tf_editor_ptr, &out_open_external_tf_editor);

                if (out_open_external_tf_editor) {
                    const auto func = [](WindowCollection::WindowConfiguration& wc) {
                        if (wc.win_callback == WindowCollection::DrawCallbacks::TRANSFER_FUNCTION) {
                            wc.win_show = true;
                        }
                    };
                    this->window_collection.EnumWindows(func);
                }
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
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    // Leave some space in histogram for text of current value
    wc.buf_current_delay += io.DeltaTime;
    int buffer_size = static_cast<int>(wc.buf_values.size());
    if (wc.ms_refresh_rate > 0.0f) {
        if (wc.buf_current_delay >= (1.0f / wc.ms_refresh_rate)) {
            if (buffer_size != wc.ms_max_history_count) {
                if (buffer_size > wc.ms_max_history_count) {
                    wc.buf_values.erase(
                        wc.buf_values.begin(), wc.buf_values.begin() + (buffer_size - wc.ms_max_history_count));

                } else if (buffer_size < wc.ms_max_history_count) {
                    wc.buf_values.insert(wc.buf_values.begin(), (wc.ms_max_history_count - buffer_size), 0.0f);
                }
            }
            if (buffer_size > 0) {
                wc.buf_values.erase(wc.buf_values.begin());
                wc.buf_values.emplace_back(io.DeltaTime * 1000.0f); // scale to milliseconds

                float max_fps = 0.0f;
                float max_ms = 0.0f;
                for (auto& v : wc.buf_values) {
                    if (v > 0.0f) {
                        max_fps = ((1.0f / v * 1000.f) > max_fps) ? (1.0f / v * 1000.f) : (max_fps);
                    }
                    max_ms = (v > max_ms) ? (v) : (max_ms);
                }

                wc.buf_plot_fps_scaling = max_fps;
                wc.buf_plot_ms_scaling = max_ms;
            }
            wc.buf_current_delay = 0.0f;
        }
    }

    // Draw window content
    if (ImGui::RadioButton("fps", (wc.ms_mode == WindowCollection::TimingModes::FPS))) {
        wc.ms_mode = WindowCollection::TimingModes::FPS;
    }
    ImGui::SameLine();

    if (ImGui::RadioButton("ms", (wc.ms_mode == WindowCollection::TimingModes::MS))) {
        wc.ms_mode = WindowCollection::TimingModes::MS;
    }

    if (this->core_instance != nullptr) {
        ImGui::TextDisabled("Frame ID:");
        ImGui::SameLine();
        ImGui::Text("%u", this->core_instance->GetFrameID());
    }

    ImGui::SameLine(
        ImGui::CalcItemWidth() - (ImGui::GetFrameHeightWithSpacing() - style.ItemSpacing.x - style.ItemInnerSpacing.x));
    if (ImGui::ArrowButton("Options_", ((wc.ms_show_options) ? (ImGuiDir_Down) : (ImGuiDir_Up)))) {
        wc.ms_show_options = !wc.ms_show_options;
    }

    std::vector<float> value_array = wc.buf_values;
    if (wc.ms_mode == WindowCollection::TimingModes::FPS) {
        for (auto& v : value_array) {
            v = (v > 0.0f) ? (1.0f / v * 1000.f) : (0.0f);
        }
    }
    float* value_ptr = (&value_array)->data();

    std::string overlay;
    if (buffer_size > 0) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(3) << value_array.back();
        overlay = stream.str();
    }

    float plot_scale_factor = 1.5f;
    if (wc.ms_mode == WindowCollection::TimingModes::FPS) {
        plot_scale_factor *= wc.buf_plot_fps_scaling;
    } else if (wc.ms_mode == WindowCollection::TimingModes::MS) {
        plot_scale_factor *= wc.buf_plot_ms_scaling;
    }

    ImGui::PlotLines(
        "###msplot", value_ptr, buffer_size, 0, overlay.c_str(), 0.0f, plot_scale_factor, ImVec2(0.0f, 50.0f));

    if (wc.ms_show_options) {
        if (ImGui::InputFloat("Refresh Rate (per sec.)", &wc.ms_refresh_rate, 1.0f, 10.0f, "%.3f",
                ImGuiInputTextFlags_EnterReturnsTrue)) {
            wc.ms_refresh_rate = std::max(1.0f, wc.ms_refresh_rate);
        }

        if (ImGui::InputInt("History Size", &wc.ms_max_history_count, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue)) {
            wc.ms_max_history_count = std::max(1, wc.ms_max_history_count);
        }

        if (ImGui::Button("Current Value")) {
#ifdef GUI_USE_GLFW
            auto glfw_win = ::glfwGetCurrentContext();
            ::glfwSetClipboardString(glfw_win, overlay.c_str());
#elif _WIN32
            ImGui::SetClipboardText(overlay.c_str());
#else // LINUX
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[GUI] Current Performance Monitor Value:\n%s", overlay.c_str());
#endif
        }
        ImGui::SameLine();

        if (ImGui::Button("All Values")) {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(3);
            auto reverse_end = value_array.rend();
            for (std::vector<float>::reverse_iterator i = value_array.rbegin(); i != reverse_end; ++i) {
                stream << (*i) << "\n";
            }
#ifdef GUI_USE_GLFW
            auto glfw_win = ::glfwGetCurrentContext();
            ::glfwSetClipboardString(glfw_win, stream.str().c_str());
#elif _WIN32
            ImGui::SetClipboardText(stream.str().c_str());
#else // LINUX
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[GUI] All Performance Monitor Values:\n%s", stream.str().c_str());
#endif
        }
        ImGui::SameLine();
        ImGui::TextUnformatted("Copy to Clipborad");
        std::string help("Values are copied in chronological order (newest first)");
        this->tooltip.Marker(help);
    }
}


void GUIWindows::drawFontWindowCallback(WindowCollection::WindowConfiguration& wc) {
    ImGuiIO& io = ImGui::GetIO();

    ImFont* font_current = ImGui::GetFont();
    if (ImGui::BeginCombo("Select available Font", font_current->GetDebugName())) {
        for (int n = this->state.graph_fonts_reserved; n < io.Fonts->Fonts.Size;
             n++) { // first fonts until index this->graph_fonts_reserved are exclusively used by configurator for the
                    // graph.
            if (ImGui::Selectable(io.Fonts->Fonts[n]->GetDebugName(), (io.Fonts->Fonts[n] == font_current)))
                io.FontDefault = io.Fonts->Fonts[n];
        }
        ImGui::EndCombo();
    }

    // Saving current font to window configuration.
    wc.font_name = std::string(font_current->GetDebugName());
    GUIUtils::Utf8Decode(wc.font_name);

    ImGui::Separator();
    ImGui::TextUnformatted("Load Font from File");
    std::string help("Same font can be loaded multiple times with different font size.");
    this->tooltip.Marker(help);

    std::string label("Font Size");
    ImGui::InputFloat(label.c_str(), &wc.buf_font_size, 1.0f, 10.0f, "%.2f", ImGuiInputTextFlags_None);
    // Validate font size
    if (wc.buf_font_size <= 0.0f) {
        wc.buf_font_size = 5.0f; // minimum valid font size
    }

    label = "Font Filename (.ttf)";
    /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
    GUIUtils::Utf8Encode(wc.buf_font_file);
    ImGui::InputText(label.c_str(), &wc.buf_font_file, ImGuiInputTextFlags_AutoSelectAll);
    GUIUtils::Utf8Decode(wc.buf_font_file);
    // Validate font file before offering load button
    if (FileUtils::FileWithExtensionExists<std::string>(wc.buf_font_file, std::string(".ttf"))) {
        if (ImGui::Button("Add Font")) {
            this->state.font_file = wc.buf_font_file;
            this->state.font_size = wc.buf_font_size;
        }
    } else {
        ImGui::TextColored(GUI_COLOR_TEXT_ERROR, "Please enter valid font filename.");
    }
}


void GUIWindows::drawMenu(void) {

    bool megamolgraph_interface = false;
    GraphPtr_t graph_ptr;
    if (this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid, graph_ptr)) {
        megamolgraph_interface = (graph_ptr->GetCoreInterface() == GraphCoreInterface::MEGAMOL_GRAPH);
    }

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
        if (ImGui::MenuItem("Exit", "ALT + 'F4'")) {
            this->triggerCoreInstanceShutdown();
            this->state.shutdown_triggered = true;
        }
        ImGui::EndMenu();
    }
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
        ImGui::Separator();
        if (ImGui::MenuItem("Reset Size and Position",
                this->hotkeys[GUIWindows::GuiHotkeyIndex::RESET_WINDOWS_POS].keycode.ToString().c_str(), nullptr)) {
            this->hotkeys[GUIWindows::GuiHotkeyIndex::RESET_WINDOWS_POS].is_pressed = true;
        }
        this->tooltip.ToolTip("Reset size and position of all windows to lie within the current viewport.");
        ImGui::EndMenu();
    }
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
    }
    if (megamolgraph_interface && (graph_ptr != nullptr)) {
        if (ImGui::BeginMenu("Render")) {
            for (auto& module_ptr : graph_ptr->GetModules()) {
                if (module_ptr->is_view) {
                    if (ImGui::MenuItem(module_ptr->FullName().c_str(), "", module_ptr->IsMainView())) {
                        if (!module_ptr->IsMainView()) {
                            // Remove all main views
                            for (auto module_ptr : graph_ptr->GetModules()) {
                                if (module_ptr->is_view && module_ptr->IsMainView()) {
                                    module_ptr->main_view_name.clear();
                                    Graph::QueueData queue_data;
                                    queue_data.name_id = module_ptr->FullName();
                                    graph_ptr->PushSyncQueue(Graph::QueueAction::REMOVE_MAIN_VIEW, queue_data);
                                }
                            }
                            // Add new main view
                            module_ptr->main_view_name = graph_ptr->GenerateUniqueMainViewName();
                            Graph::QueueData queue_data;
                            queue_data.name_id = module_ptr->FullName();
                            graph_ptr->PushSyncQueue(Graph::QueueAction::CREATE_MAIN_VIEW, queue_data);
                        } else {
                            module_ptr->main_view_name.clear();
                            Graph::QueueData queue_data;
                            queue_data.name_id = module_ptr->FullName();
                            graph_ptr->PushSyncQueue(Graph::QueueAction::REMOVE_MAIN_VIEW, queue_data);
                        }
                    }
                }
            }
            if (ImGui::MenuItem("Toggle Main Views",
                    this->hotkeys[GUIWindows::GuiHotkeyIndex::TOGGLE_MAIN_VIEWS].keycode.ToString().c_str())) {
                this->state.toggle_main_view = true;
            }
            ImGui::EndMenu();
        }
    }
    if (ImGui::BeginMenu("Settings")) {
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
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help")) {
        if (ImGui::MenuItem("About")) {
            this->state.open_popup_about = true;
        }
        ImGui::EndMenu();
    }
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
#ifdef GUI_USE_GLFW
            auto glfw_win = ::glfwGetCurrentContext();
            ::glfwSetClipboardString(glfw_win, email.c_str());
#elif _WIN32
            ImGui::SetClipboardText(email.c_str());
#else // LINUX
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] E-Mail address:\n%s", email.c_str());
#endif
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(mailstr.c_str());

        if (ImGui::Button("Copy Website")) {
#ifdef GUI_USE_GLFW
            auto glfw_win = ::glfwGetCurrentContext();
            ::glfwSetClipboardString(glfw_win, web_link.c_str());
#elif _WIN32
            ImGui::SetClipboardText(web_link.c_str());
#else // LINUX
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Website link:\n%s", web_link.c_str());
#endif
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(webstr.c_str());

        if (ImGui::Button("Copy GitHub###megamol_copy_github")) {
#ifdef GUI_USE_GLFW
            auto glfw_win = ::glfwGetCurrentContext();
            ::glfwSetClipboardString(glfw_win, github_link.c_str());
#elif _WIN32
            ImGui::SetClipboardText(github_link.c_str());
#else // LINUX
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] GitHub link:\n%s", github_link.c_str());
#endif
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(gitstr.c_str());

        ImGui::Separator();
        ImGui::TextUnformatted(imguistr.c_str());
        if (ImGui::Button("Copy GitHub###imgui_copy_github")) {
#ifdef GUI_USE_GLFW
            auto glfw_win = ::glfwGetCurrentContext();
            ::glfwSetClipboardString(glfw_win, imgui_link.c_str());
#elif _WIN32
            ImGui::SetClipboardText(imgui_link.c_str());
#else // LINUX
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] ImGui GitHub Link:\n%s", imgui_link.c_str());
#endif
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(imguigitstr.c_str());

        ImGui::Separator();
        ImGui::TextUnformatted(about.c_str());

        ImGui::Separator();
        if (ImGui::Button("Close")) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    // Save project pop-up
    this->state.open_popup_save |= this->hotkeys[GUIWindows::GuiHotkeyIndex::SAVE_PROJECT].is_pressed;
    bool confirmed, aborted;
    bool popup_failed = false;
    std::string filename;
    GraphPtr_t graph_ptr;
    if (this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid, graph_ptr)) {
        filename = graph_ptr->GetFilename();
    }
    if (graph_ptr != nullptr) {
        this->state.open_popup_save |= this->configurator.ConsumeTriggeredGlobalProjectSave();
        if (this->file_browser.PopUp(
                FileBrowserWidget::FileBrowserFlag::SAVE, "Save Project", this->state.open_popup_save, filename)) {

            graph_ptr->SetFilename(filename);
            popup_failed |= !this->configurator.GetGraphCollection().SaveProjectToFile(
                this->state.graph_uid, filename, this->dump_state_to_file(filename));
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
        if (this->file_browser.PopUp(
                FileBrowserWidget::FileBrowserFlag::LOAD, "Load Project", this->state.open_popup_load, filename)) {
            graph_ptr->Clear();
            popup_failed |= (GUI_INVALID_ID == this->configurator.GetGraphCollection().LoadAddProjectFromFile(
                                                   this->state.graph_uid, filename));
        }
        MinimalPopUp::PopUp("Failed to Load Project", popup_failed, "See console log output for more information.", "",
            confirmed, "Cancel", aborted);
    }
    this->state.open_popup_load = false;
    this->hotkeys[GUIWindows::GuiHotkeyIndex::LOAD_PROJECT].is_pressed = false;

    // File name for screenshot pop-up
    if (this->file_browser.PopUp(FileBrowserWidget::FileBrowserFlag::SAVE, "Select Filename for Screenshot",
            this->state.open_popup_screenshot, this->state.screenshot_filepath, ".png")) {
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
    bool window_maximized = ((wc.win_size.x == window_viewport.x) && (wc.win_size.y == window_viewport.y));
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
            wc.win_reset = true;
        }
        ImGui::Separator();

/// DOCKING
#if (defined(IMGUI_HAS_VIEWPORT) && defined(IMGUI_HAS_DOCK))
        ImGui::MenuItem("Docking", "Shift + Left-Drag", false, false);
#else
        ImGui::MenuItem("Docking", nullptr, false, false);
#endif
        if (ImGui::ArrowButton("dock_left", ImGuiDir_Left)) {
            wc.win_position.x = 0.0f;
            wc.win_reset = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::ArrowButton("dock_up", ImGuiDir_Up)) {
            wc.win_position.y = 0.0f;
            wc.win_reset = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::ArrowButton("dock_down", ImGuiDir_Down)) {
            wc.win_position.y = viewport.y - wc.win_size.y;
            wc.win_reset = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::ArrowButton("dock_right", ImGuiDir_Right)) {
            wc.win_position.x = viewport.x - wc.win_size.x;
            wc.win_reset = true;
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
            wc.win_reset = true;
        } else {
            // Window is minimized
            ImVec2 window_viewport = ImVec2(viewport.x, viewport.y - y_offset);
            wc.win_reset_size = wc.win_size;
            wc.win_reset_position = wc.win_position;
            wc.win_size = window_viewport;
            wc.win_position = ImVec2(0.0f, y_offset);
            wc.win_reset = true;
        }
    }

    // Move current window below window menu
    if (wc.win_soft_reset || wc.win_reset || (this->state.menu_visible && ImGui::IsMouseReleased(0))) {
        float y_offset = ImGui::GetFrameHeight();
        if (wc.win_position.y < y_offset) {
            wc.win_position.y = y_offset;
            ImGui::SetWindowPos(wc.win_position, ImGuiCond_Always);
        }
    }

    // Apply soft reset of window reset position and reset size
    if (wc.win_soft_reset) {
        this->window_collection.SoftResetWindowSizePosition(wc);
        wc.win_soft_reset = false;
    }

    // Apply window position and size reset
    if (wc.win_reset) {
        this->window_collection.ResetWindowSizePosition(wc);
        wc.win_reset = false;
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

        GraphPtr_t graph_ptr;
        if (this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid, graph_ptr)) {
            for (auto& module_ptr : graph_ptr->GetModules()) {
                for (auto& param : module_ptr->parameters) {

                    if (param.type == Param_t::BUTTON) {
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
                                keyCode.ToString().c_str(), param.full_name.c_str());
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

    if (this->core_instance != nullptr) {
#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Shutdown MegaMol instance.");
#endif // GUI_VERBOSE
        this->core_instance->Shutdown();
    }
}


std::string megamol::gui::GUIWindows::dump_state_to_file(const std::string& filename) {

    nlohmann::json state_json;

    if (this->state_to_json(state_json)) {
        std::string state_str = state_json.dump(); // No line feed
        return state_str;
    }
    return std::string("");
}


bool megamol::gui::GUIWindows::load_state_from_file(const std::string& filename) {

    std::string state_str;
    if (FileUtils::ReadFile(filename, state_str, true)) {
        state_str = GUIUtils::ExtractGUIState(state_str);
        if (state_str.empty())
            return false;
        nlohmann::json in_json = nlohmann::json::parse(state_str);
        return this->state_from_json(in_json);
    }

    return false;
}


bool megamol::gui::GUIWindows::state_from_json(const nlohmann::json& in_json) {

    try {
        if (!in_json.is_object()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Invalid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        // Read GUI state
        for (auto& header_item : in_json.items()) {
            if (header_item.key() == GUI_JSON_TAG_GUI) {
                auto gui_state = header_item.value();
                megamol::core::utility::get_json_value<bool>(gui_state, {"menu_visible"}, &this->state.menu_visible);
                int style = 0;
                megamol::core::utility::get_json_value<int>(gui_state, {"style"}, &style);
                this->state.style = static_cast<GUIWindows::Styles>(style);
                this->state.style_changed = true;
            }
        }

        // Read window configurations
        this->window_collection.StateFromJSON(in_json);

        // Read configurator state
        this->configurator.StateFromJSON(in_json);

        // Read GUI state of parameters (groups)
        GraphPtr_t graph_ptr;
        if (this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid, graph_ptr)) {
            for (auto& module_ptr : graph_ptr->GetModules()) {
                std::string module_full_name = module_ptr->FullName();
                // Parameter Groups
                module_ptr->present.param_groups.StateFromJSON(in_json, module_full_name);
                // Parameters
                for (auto& param : module_ptr->parameters) {
                    std::string param_full_name = module_full_name + "::" + param.full_name;
                    param.present.StateFromJSON(in_json, param_full_name);
                    param.present.ForceSetGUIStateDirty();
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


bool megamol::gui::GUIWindows::state_to_json(nlohmann::json& inout_json) {

    try {
        // Write GUI state
        inout_json[GUI_JSON_TAG_GUI]["menu_visible"] = this->state.menu_visible;
        inout_json[GUI_JSON_TAG_GUI]["style"] = static_cast<int>(this->state.style);

        // Write window configuration
        this->window_collection.StateToJSON(inout_json);

        // Write the configurator state
        this->configurator.StateToJSON(inout_json);

        // Write GUI state of parameters (groups)
        GraphPtr_t graph_ptr;
        if (this->configurator.GetGraphCollection().GetGraph(this->state.graph_uid, graph_ptr)) {
            for (auto& module_ptr : graph_ptr->GetModules()) {
                std::string module_full_name = module_ptr->FullName();
                // Parameter Groups
                module_ptr->present.param_groups.StateToJSON(inout_json, module_full_name);
                // Parameters
                for (auto& param : module_ptr->parameters) {
                    std::string param_full_name = module_full_name + "::" + param.full_name;
                    param.present.StateToJSON(inout_json, param_full_name);
                }
            }
        }

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

    this->state.gui_enabled = true;
    this->state.enable_gui_post = true;
    this->state.style = GUIWindows::Styles::DarkColors;
    this->state.style_changed = true;
    this->state.autosave_gui_state = false;
    this->state.project_script_paths.clear();
    this->state.graph_uid = GUI_INVALID_ID;
    this->state.font_file = "";
    this->state.font_size = 13.0f;
    this->state.font_index = GUI_INVALID_ID;
    this->state.font_utf8_ranges.clear();
    this->state.win_save_state = false;
    this->state.win_save_delay = 0.0f;
    this->state.win_delete = "";
    this->state.last_instance_time = 0.0;
    this->state.open_popup_about = false;
    this->state.open_popup_save = false;
    this->state.open_popup_load = false;
    this->state.open_popup_screenshot = false;
    this->state.menu_visible = true;
    this->state.graph_fonts_reserved = 0;
    this->state.toggle_main_view = false;
    this->state.shutdown_triggered = false;
    this->state.screenshot_triggered = false;
    this->state.screenshot_filepath = "megamol_screenshot.png";
    this->state.screenshot_filepath_id = 0;
    this->state.last_script_filename = "";
    this->state.hotkeys_check_once = true;

    this->create_not_existing_png_filepath(this->state.screenshot_filepath);
}


bool megamol::gui::GUIWindows::create_not_existing_png_filepath(std::string& inout_filepath) {

    // Check for existing file
    bool created_filepath = false;
    if (!inout_filepath.empty()) {
        while (FileUtils::FileExists<std::string>(inout_filepath)) {
            // Create new filename with iterating suffix
            std::string filename = FileUtils::GetFilenameStem<std::string>(inout_filepath);
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
