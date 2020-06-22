/*
 * GUIWindows.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

/**
 * USED HOTKEYS:
 *
 * - Show/hide Menu:            F12
 * - Show/hide Windows:         F7-F11
 * - Reset Windows:             Shift + F7-F11
 * - Search Parameter:          Ctrl  + p
 * - Save Running Project:      Ctrl  + s
 * - Quit Program:              Alt   + F4
 */

#include "stdafx.h"
#include "GUIWindows.h"


using namespace megamol;
using namespace megamol::gui;


GUIWindows::GUIWindows()
    : core_instance(nullptr)
    , param_slots()
    , style_param("style", "Color style, theme")
    , state_param(GUI_GUI_STATE_PARAM_NAME, "Current state of all windows.")
    , autostart_configurator("autostart_configurator", "Start the configurator at start up automatically. ")
    , context(nullptr)
    , impl(Implementation::NONE)
    , window_manager()
    , tf_editor_ptr(nullptr)
    , configurator()
    , utils()
    , file_utils()
    , state()
    , parent_module_fullname()
    , graph_fonts_reserved(0)
    , graph_uid(GUI_INVALID_ID)
    , graph_manager()
    , param_core_interface_map() {

    core::param::EnumParam* styles = new core::param::EnumParam((int)(Styles::DarkColors));
    styles->SetTypePair(Styles::CorporateGray, "Corporate Gray");
    styles->SetTypePair(Styles::CorporateWhite, "Corporate White");
    styles->SetTypePair(Styles::DarkColors, "Dark Colors");
    styles->SetTypePair(Styles::LightColors, "Light Colors");
    this->style_param << styles;
    this->style_param.ForceSetDirty();
    styles = nullptr;

    this->state_param << new core::param::StringParam("");
    this->state_param.Parameter()->SetGUIVisible(false);
    this->state_param.Parameter()->SetGUIReadOnly(true);

    this->autostart_configurator << new core::param::BoolParam(false);

    this->param_slots.clear();
    this->param_slots.push_back(&this->state_param);
    this->param_slots.push_back(&this->style_param);
    this->param_slots.push_back(&this->autostart_configurator);
    for (auto& configurator_param : this->configurator.GetParams()) {
        this->param_slots.push_back(configurator_param);
    }

    this->hotkeys[GUIWindows::GuiHotkeyIndex::EXIT_PROGRAM] = megamol::gui::HotkeyDataType(
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F4, core::view::Modifier::ALT), false);
    this->hotkeys[GUIWindows::GuiHotkeyIndex::PARAMETER_SEARCH] = megamol::gui::HotkeyDataType(
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_P, core::view::Modifier::CTRL), false);
    this->hotkeys[GUIWindows::GuiHotkeyIndex::SAVE_PROJECT] = megamol::gui::HotkeyDataType(
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_S, core::view::Modifier::CTRL), false);
    this->hotkeys[GUIWindows::GuiHotkeyIndex::MENU] = megamol::gui::HotkeyDataType(
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F12, core::view::Modifier::NONE), false);

    this->tf_editor_ptr = std::make_shared<TransferFunctionEditor>();
}


GUIWindows::~GUIWindows() { this->destroyContext(); }


bool GUIWindows::CreateContext_GL(megamol::core::CoreInstance* instance) {

    if (instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to core instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->core_instance = instance;

    if (this->createContext()) {
        // Init OpenGL for ImGui
        const char* glsl_version = "#version 130"; /// "#version 150"
        if (ImGui_ImplOpenGL3_Init(glsl_version)) {
            this->impl = Implementation::OpenGL;
            return true;
        }
    }

    return false;
}


bool GUIWindows::PreDraw(
    const std::string& module_fullname, vislib::math::Rectangle<int> viewport, double instanceTime) {

    if (this->impl == Implementation::NONE) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Found no initialized ImGui implementation. First call CreateContext_XXX() once. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->context == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Found no valid ImGui context. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->core_instance == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to core instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }

    // Set ImGui context
    ImGui::SetCurrentContext(this->context);
    this->core_instance->SetCurrentImGuiContext(this->context);

    // Loading and/or updating currently running core graph
    /// TODO Update
    /// Run only once for now ....
    if (this->param_core_interface_map.empty()) {
        this->graph_manager.LoadCallStock(core_instance);
        this->graph_uid = this->graph_manager.LoadUpdateProjectFromCore(
            this->graph_uid, this->core_instance, this->param_core_interface_map);
    }

    // Checking global hotkeys
    if (std::get<1>(this->hotkeys[GUIWindows::GuiHotkeyIndex::EXIT_PROGRAM])) {
        this->shutdown();
        return true;
    }
    if (std::get<1>(this->hotkeys[GUIWindows::GuiHotkeyIndex::MENU])) {
        this->state.menu_visible = !this->state.menu_visible;
        std::get<1>(this->hotkeys[GUIWindows::GuiHotkeyIndex::MENU]) = false;
    }
    this->validateParameters();
    this->checkMultipleHotkeyAssignement();
    this->parent_module_fullname = module_fullname;

    auto viewportWidth = viewport.Width();
    auto viewportHeight = viewport.Height();

    // Set IO stuff for next frame --------------------------------------------
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)viewportWidth, (float)viewportHeight);
    io.DisplayFramebufferScale = ImVec2(1.0, 1.0);

    if ((instanceTime - this->state.last_instance_time) < 0.0) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Current instance time results in negative time delta. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
    }
    io.DeltaTime = ((instanceTime - this->state.last_instance_time) > 0.0)
                       ? (static_cast<float>(instanceTime - this->state.last_instance_time))
                       : (io.DeltaTime);
    this->state.last_instance_time = ((instanceTime - this->state.last_instance_time) > 0.0)
                                         ? (instanceTime)
                                         : (this->state.last_instance_time + io.DeltaTime);

    // Changes that need to be applied before next frame ----------------------
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
    if (this->state.font_index >= this->graph_fonts_reserved) {
        if (this->state.font_index < static_cast<unsigned int>(io.Fonts->Fonts.Size)) {
            io.FontDefault = io.Fonts->Fonts[this->state.font_index];
        }
        this->state.font_index = GUI_INVALID_ID;
    }

    // Deleting window
    if (!this->state.win_delete.empty()) {
        this->window_manager.DeleteWindowConfiguration(this->state.win_delete);
        this->state.win_delete.clear();
    }

    // Start new ImGui frame --------------------------------------------------
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();

    return true;
}


bool GUIWindows::PostDraw(void) {

    if (this->impl == Implementation::NONE) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Found no initialized ImGui implementation. First call CreateContext_XXX() once. [%s, %s, line %d]\n",
            __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    if (ImGui::GetCurrentContext() != this->context) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "Unknown ImGui context ... [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    ImGui::SetCurrentContext(this->context);
    if (this->context == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Found no valid ImGui context. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    ImGuiIO& io = ImGui::GetIO();
    ImVec2 viewport = ImVec2(io.DisplaySize.x, io.DisplaySize.y);

    // Main Menu ---------------------------------------------------------------
    if (this->state.menu_visible) {
        if (ImGui::BeginMainMenuBar()) {
            this->drawMenu();
            ImGui::EndMainMenuBar();
        }
    }

    // Draw Windows ------------------------------------------------------------
    const auto func = [&, this](WindowManager::WindowConfiguration& wc) {
        // Loading changed window state of font (even if window is not shown)
        if ((wc.win_callback == WindowManager::DrawCallbacks::FONT) && wc.buf_font_reset) {
            if (!wc.font_name.empty()) {
                this->state.font_index = GUI_INVALID_ID;
                for (unsigned int n = this->graph_fonts_reserved; n < static_cast<unsigned int>(io.Fonts->Fonts.Size);
                     n++) {
                    std::string font_name = std::string(io.Fonts->Fonts[n]->GetDebugName());
                    GUIUtils::Utf8Decode(font_name);
                    if (font_name == wc.font_name) {
                        this->state.font_index = n;
                    }
                }
                if (this->state.font_index == GUI_INVALID_ID) {
                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "Could not find font '%s' for loaded state. [%s, %s, line %d]\n", wc.font_name.c_str(),
                        __FILE__, __FUNCTION__, __LINE__);
                }
            }
            wc.buf_font_reset = false;
        }
        // Loading changed window state of transfer function editor (even if window is not shown)
        if ((wc.win_callback == WindowManager::DrawCallbacks::TRANSFER_FUNCTION) && wc.buf_tfe_reset) {
            this->tf_editor_ptr->SetMinimized(wc.tfe_view_minimized);
            this->tf_editor_ptr->SetVertical(wc.tfe_view_vertical);
            for (auto& pair : this->param_core_interface_map) {
                auto param_ptr = pair.second;
                if (param_ptr != nullptr) {
                    if ((wc.tfe_active_param == param_ptr->full_name) &&
                        (param_ptr->type == ParamType::TRANSFERFUNCTION)) {
                        this->tf_editor_ptr->SetConnectedParameter(param_ptr);
                        this->tf_editor_ptr->SetTransferFunction(std::get<std::string>(param_ptr->GetValue()), true);
                    }
                }
            }
            wc.buf_tfe_reset = false;
        }

        // Draw window content
        if (wc.win_show) {
            ImGui::SetNextWindowBgAlpha(1.0f);

            // Change window flags depending on current view of transfer function editor
            if (wc.win_callback == WindowManager::DrawCallbacks::TRANSFER_FUNCTION) {
                if (this->tf_editor_ptr->IsMinimized()) {
                    wc.win_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize |
                                   ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar;
                } else {
                    wc.win_flags = ImGuiWindowFlags_AlwaysAutoResize;
                }
                wc.tfe_view_minimized = this->tf_editor_ptr->IsMinimized();
                wc.tfe_view_vertical = this->tf_editor_ptr->IsVertical();
            }

            // Begin Window
            if (!ImGui::Begin(wc.win_name.c_str(), &wc.win_show, wc.win_flags)) {
                ImGui::End(); // early ending
                return;
            }

            // Always set configurator window size to current viewport
            if (wc.win_callback == WindowManager::DrawCallbacks::CONFIGURATOR) {
                float y_offset = (this->state.menu_visible) ? (ImGui::GetFrameHeight()) : (0.0f);
                wc.win_size = ImVec2(viewport.x, viewport.y - y_offset);
                wc.win_position = ImVec2(0.0f, y_offset);
                wc.win_reset = true;
            }

            // Apply soft reset of window position and size (before calling window callback)
            if (wc.win_soft_reset) {
                this->window_manager.SoftResetWindowSizePos(wc);
                wc.win_soft_reset = false;
            }

            // Force window menu
            if (this->state.menu_visible && ImGui::IsMouseReleased(0)) {
                float y_offset = ImGui::GetFrameHeight();
                if (wc.win_position.y < y_offset) {
                    wc.win_position.y = y_offset;
                    wc.win_reset = true;
                }
            }
            // Apply window position and size reset (before calling window callback)
            if (wc.win_reset) {
                this->window_manager.ResetWindowPosSize(wc);
                wc.win_reset = false;
            }

            // Calling callback drawing window content
            auto cb = this->window_manager.WindowCallback(wc.win_callback);
            if (cb) {
                cb(wc);
            } else {
                vislib::sys::Log::DefaultLog.WriteError(
                    "Missing valid callback for WindowDrawCallback: '%d'. [%s, %s, line %d]\n", (int)wc.win_callback,
                    __FILE__, __FUNCTION__, __LINE__);
            }

            // Saving current window position and size for all window configurations for possible state saving.
            wc.win_position = ImGui::GetWindowPos();
            wc.win_size = ImGui::GetWindowSize();

            ImGui::End();
        }
    };
    this->window_manager.EnumWindows(func);

    // Draw global parameter widgets -------------------------------------------
    for (auto& pair : this->param_core_interface_map) {
        this->drawParameter(pair.second, configurator::ParameterPresentation::WidgetScope::GLOBAL);
    }

    // Synchronizing parameter values -----------------------------------------
    if (this->core_instance != nullptr) {
        this->core_instance->EnumParameters([&, this](const auto& mod, auto& slot) {
            auto parameter_slot_ptr = slot.Parameter();
            if (parameter_slot_ptr.IsNull()) {
                return;
            }
            auto param_ref = &(*parameter_slot_ptr);
            auto param_ptr = this->param_core_interface_map[param_ref];
            if (param_ptr == nullptr) return;
            /// TODO GUI changes (visibility, readonly, presentation) are only propagates to core on value change.
            if (param_ptr->IsDirty()) {
                megamol::gui::configurator::WriteCoreParameter((*param_ptr), slot);
                param_ptr->ResetDirty();
            } else {
                megamol::gui::configurator::ReadCoreParameter(slot, (*param_ptr), false);
            }
        });
    } else {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to core instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }

    // Draw pop-ups ------------------------------------------------------------
    this->drawPopUps();

    // Render the current ImGui frame ------------------------------------------
    glViewport(0, 0, static_cast<GLsizei>(viewport.x), static_cast<GLsizei>(viewport.y));
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    // Reset hotkeys
    for (auto& h : this->hotkeys) {
        std::get<1>(h) = false;
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
        if (this->isHotkeyPressed(std::get<0>(h))) {
            std::get<1>(h) = true;
            hotkeyPressed = true;
        }
    }
    // Configurator
    for (auto& h : this->configurator.GetHotkeys()) {
        if (this->isHotkeyPressed(std::get<0>(h))) {
            std::get<1>(h) = true;
            hotkeyPressed = true;
        }
    }
    if (hotkeyPressed) return true;

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
    const auto windows_func = [&](WindowManager::WindowConfiguration& wc) {
        bool windowHotkeyPressed = this->isHotkeyPressed(wc.win_hotkey);
        if (windowHotkeyPressed) {
            wc.win_show = !wc.win_show;
        }
        hotkeyPressed = (hotkeyPressed || windowHotkeyPressed);

        auto window_hotkey = wc.win_hotkey;
        auto mods = window_hotkey.mods;
        mods |= megamol::core::view::Modifier::SHIFT;
        window_hotkey = megamol::core::view::KeyCode(window_hotkey.key, mods);
        windowHotkeyPressed = this->isHotkeyPressed(window_hotkey);
        if (windowHotkeyPressed) {
            wc.win_soft_reset = true;
        }
        hotkeyPressed = (hotkeyPressed || windowHotkeyPressed);
    };
    this->window_manager.EnumWindows(windows_func);
    if (hotkeyPressed) return true;

    // Always consume keyboard input if requested by any imgui widget (e.g. text input).
    // User expects hotkey priority of text input thus needs to be processed before parameter hotkeys.
    if (io.WantTextInput) { /// io.WantCaptureKeyboard
        return true;
    }

    // Check for parameter hotkeys
    /// TODO Fix. Use different lists for new parameter windows and module filter.
    /*
    std::vector<std::string> modules_list;
    const auto modfunc = [&](WindowManager::WindowConfiguration& wc) {
        for (auto& m : wc.param_modules_list) {
            modules_list.emplace_back(m);
        }
    };
    this->window_manager.EnumWindows(modfunc);

    bool consider_module = false;
    std::string current_module_fullname = "";
    */
    hotkeyPressed = false;
    for (auto& pair : this->param_core_interface_map) {
        /*
        std::string module_fullname = pair.second->GetNameSpace();
        if (current_module_fullname != module_fullname) {
            current_module_fullname = module_fullname;
            consider_module = this->considerModule(module_fullname, modules_list);
        }
        if (consider_module) {
        */
        auto param_ptr = pair.second;
        if (param_ptr == nullptr) continue;
        if (param_ptr->type == ParamType::BUTTON) {
            auto keyCode = param_ptr->GetStorage<megamol::core::view::KeyCode>();
            if (this->isHotkeyPressed(keyCode)) {
                param_ptr->ForceSetDirty();
                hotkeyPressed = true;
                // Break loop after first occurrence of parameter hotkey
                break;
            }
        }
        //}
    }

    return hotkeyPressed;
}


bool GUIWindows::OnChar(unsigned int codePoint) {
    ImGui::SetCurrentContext(this->context);

    ImGuiIO& io = ImGui::GetIO();
    io.ClearInputCharacters();
    if (codePoint > 0 && codePoint < 0x10000) {
        io.AddInputCharacter((unsigned short)codePoint);
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

    /* DISBALED --- Is it really neccessary to serialze GUI state after every change?
    // Trigger saving state when mouse hovered any window and on button mouse release event
    if (ImGui::IsMouseReleased[buttonIndex] && hoverFlags) {
        this->state.win_save_state = true;
        this->state.win_save_delay = 0.0f;
    }
    */

    io.MouseDown[buttonIndex] = down;

    // Always consumed if any imgui windows is hovered.
    bool consumed = ImGui::IsWindowHovered(hoverFlags);
    return consumed;
}


bool GUIWindows::OnMouseScroll(double dx, double dy) {
    ImGui::SetCurrentContext(this->context);

    ImGuiIO& io = ImGui::GetIO();
    io.MouseWheelH += (float)dx;
    io.MouseWheel += (float)dy;

    auto hoverFlags = ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenDisabled |
                      ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem;

    // Always consumed if any imgui windows is hovered.
    bool consumed = ImGui::IsWindowHovered(hoverFlags);
    return consumed;
}


bool GUIWindows::createContext(void) {

    // Check for successfully created tf editor
    if (this->tf_editor_ptr == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to transfer function editor is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    // Create ImGui context ---------------------------------------------------
    // Check for existing context and share FontAtlas with new context (required by ImGui).
    bool other_context = (ImGui::GetCurrentContext() != nullptr);
    ImFontAtlas* current_fonts = nullptr;
    if (other_context) {
        ImGuiIO& current_io = ImGui::GetIO();
        current_fonts = current_io.Fonts;
    }
    IMGUI_CHECKVERSION();
    this->context = ImGui::CreateContext(current_fonts);
    if (this->context == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Unable to create ImGui context. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    ImGui::SetCurrentContext(this->context);

    // Register window callbacks in window manager ----------------------------
    this->window_manager.RegisterDrawWindowCallback(WindowManager::DrawCallbacks::MAIN_PARAMETERS,
        [&, this](WindowManager::WindowConfiguration& wc) { this->drawParamWindowCallback(wc); });
    this->window_manager.RegisterDrawWindowCallback(WindowManager::DrawCallbacks::PARAMETERS,
        [&, this](WindowManager::WindowConfiguration& wc) { this->drawParamWindowCallback(wc); });
    this->window_manager.RegisterDrawWindowCallback(WindowManager::DrawCallbacks::PERFORMANCE,
        [&, this](WindowManager::WindowConfiguration& wc) { this->drawFpsWindowCallback(wc); });
    this->window_manager.RegisterDrawWindowCallback(WindowManager::DrawCallbacks::FONT,
        [&, this](WindowManager::WindowConfiguration& wc) { this->drawFontWindowCallback(wc); });
    this->window_manager.RegisterDrawWindowCallback(WindowManager::DrawCallbacks::TRANSFER_FUNCTION,
        [&, this](WindowManager::WindowConfiguration& wc) { this->drawTransferFunctionWindowCallback(wc); });
    this->window_manager.RegisterDrawWindowCallback(WindowManager::DrawCallbacks::CONFIGURATOR,
        [&, this](WindowManager::WindowConfiguration& wc) { this->drawConfiguratorWindowCallback(wc); });

    // Create window configurations
    WindowManager::WindowConfiguration buf_win;
    buf_win.win_reset = true;
    buf_win.win_position = ImVec2(0.0f, 0.0f);
    buf_win.win_size = ImVec2(400.0f, 600.0f);

    // MAIN Window ------------------------------------------------------------
    buf_win.win_name = "All Parameters";
    buf_win.win_show = false;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F11);
    buf_win.win_flags = ImGuiWindowFlags_HorizontalScrollbar;
    buf_win.win_callback = WindowManager::DrawCallbacks::MAIN_PARAMETERS;
    buf_win.win_reset_size = buf_win.win_size;
    this->window_manager.AddWindowConfiguration(buf_win);

    // FPS/MS Window ----------------------------------------------------------
    buf_win.win_name = "Performance Metrics";
    buf_win.win_show = false;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F10);
    buf_win.win_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
    buf_win.win_callback = WindowManager::DrawCallbacks::PERFORMANCE;
    this->window_manager.AddWindowConfiguration(buf_win);

    // FONT Window ------------------------------------------------------------
    buf_win.win_name = "Font Settings";
    buf_win.win_show = false;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F9);
    buf_win.win_flags = ImGuiWindowFlags_AlwaysAutoResize;
    buf_win.win_callback = WindowManager::DrawCallbacks::FONT;
    this->window_manager.AddWindowConfiguration(buf_win);

    // TRANSFER FUNCTION Window -----------------------------------------------
    buf_win.win_name = "Transfer Function Editor";
    buf_win.win_show = false;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F8);
    buf_win.win_flags = ImGuiWindowFlags_AlwaysAutoResize;
    buf_win.win_callback = WindowManager::DrawCallbacks::TRANSFER_FUNCTION;
    this->window_manager.AddWindowConfiguration(buf_win);

    // CONFIGURATOR Window -----------------------------------------------
    buf_win.win_name = "Configurator";
    buf_win.win_show = false;
    // State of configurator should not be stored (visibility is configured via auto load parameter and will always be
    // viewport size).
    buf_win.win_store_config = false;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F7);
    buf_win.win_flags =
        ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;
    buf_win.win_callback = WindowManager::DrawCallbacks::CONFIGURATOR;
    // buf_win.win_size is set to current viewport later
    this->window_manager.AddWindowConfiguration(buf_win);

    // Style settings ---------------------------------------------------------
    ImGui::SetColorEditOptions(ImGuiColorEditFlags_Uint8 | ImGuiColorEditFlags_DisplayRGB |
                               ImGuiColorEditFlags_PickerHueWheel | ImGuiColorEditFlags_AlphaBar |
                               ImGuiColorEditFlags_AlphaPreview);
    /// ... for detailed settings see styles defined in separate headers.
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;

    // IO settings ------------------------------------------------------------
    ImGuiIO& io = ImGui::GetIO();
    io.IniSavingRate = 5.0f;                              //  in seconds
    io.IniFilename = nullptr;                             // "imgui.ini"; - disabled, using own window settings profile
    io.LogFilename = "imgui_log.txt";                     // (set to nullptr to disable)
    io.FontAllowUserScaling = false;                      // disable font scaling using ctrl + mouse wheel
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // allow keyboard navigation

    // Init global state -------------------------------------------------------
    this->state.font_file = "";
    this->state.font_size = 13.0f;
    this->state.font_index = GUI_INVALID_ID;
    this->state.win_save_state = false;
    this->state.win_save_delay = 0.0f;
    this->state.win_delete = "";
    this->state.last_instance_time = 0.0f;
    this->state.open_popup_about = false;
    this->state.open_popup_save = false;
    this->state.project_file = "";
    this->state.menu_visible = true;
    this->state.hotkeys_check_once = true;
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
    if (!other_context) {
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
            const vislib::Array<vislib::StringW>& search_paths =
                this->core_instance->Configuration().ResourceDirectories();
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
        } else {
            vislib::sys::Log::DefaultLog.WriteError(
                "Pointer to core instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
        // Configurator Graph Font: Add default font at first n indices for exclusive use in configurator graph.
        /// Workaround: Using different font sizes for different graph zooming factors to improve font readability when
        /// zooming.
        const auto graph_font_scalings = this->configurator.GetGraphFontScalings();
        this->graph_fonts_reserved = graph_font_scalings.size();
        if (configurator_font.empty()) {
            for (unsigned int i = 0; i < this->graph_fonts_reserved; i++) {
                io.Fonts->AddFontDefault(&config);
            }
        } else {
            for (unsigned int i = 0; i < this->graph_fonts_reserved; i++) {
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

    this->core_instance = nullptr;

    if (this->impl != Implementation::NONE) {
        if (this->context != nullptr) {
            switch (this->impl) {
            case (Implementation::OpenGL):
                ImGui_ImplOpenGL3_Shutdown();
                break;
            default:
                break;
            }
            ImGui::DestroyContext(this->context);
        }
    }
    this->impl = Implementation::NONE;

    return true;
}


void GUIWindows::validateParameters() {
    if (this->style_param.IsDirty()) {
        auto style = static_cast<Styles>(this->style_param.Param<core::param::EnumParam>()->Value());
        switch (style) {
        case Styles::CorporateGray:
            CorporateGreyStyle();
            break;
        case Styles::CorporateWhite:
            CorporateWhiteStyle();
            break;
        case Styles::DarkColors:
            ImGui::StyleColorsDark();
            break;
        case Styles::LightColors:
            ImGui::StyleColorsLight();
            break;
        }
        this->style_param.ResetDirty();
    }

    if (this->state_param.IsDirty()) {
        std::string state = std::string(this->state_param.Param<core::param::StringParam>()->Value().PeekBuffer());
        this->window_manager.StateFromJsonString(state);
        this->gui_and_parameters_state_from_json_string(state);
        this->state_param.ResetDirty();
    }
    /* DISBALED --- Is it really neccessary to serialze GUI state after every change?
    ImGuiIO& io = ImGui::GetIO();
    this->state.win_save_delay += io.DeltaTime;
    else if (this->state.win_save_state && (this->state.win_save_delay > 1.0f)) {
        // Delayed saving after triggering saving state (in seconds).
        this->save_state_to_parameter();
        this->state.win_save_state = false;
    }
    */

    if (this->autostart_configurator.IsDirty()) {
        bool autostart = this->autostart_configurator.Param<core::param::BoolParam>()->Value();
        if (autostart) {
            const auto configurator_func = [](WindowManager::WindowConfiguration& wc) {
                if (wc.win_callback == WindowManager::DrawCallbacks::CONFIGURATOR) {
                    wc.win_show = true;
                }
            };
            this->window_manager.EnumWindows(configurator_func);
        }
        this->autostart_configurator.ResetDirty();
    }
}


void GUIWindows::drawTransferFunctionWindowCallback(WindowManager::WindowConfiguration& wc) {

    this->tf_editor_ptr->Draw(true);

    auto param_ptr = this->tf_editor_ptr->GetConnectedParameter();
    if (param_ptr != nullptr) {
        wc.tfe_active_param = param_ptr->full_name;
    }
}


void GUIWindows::drawConfiguratorWindowCallback(WindowManager::WindowConfiguration& wc) {

    this->configurator.Draw(wc, this->core_instance);
}


void GUIWindows::drawParamWindowCallback(WindowManager::WindowConfiguration& wc) {

    // Mode
    ImGui::BeginGroup();
    this->utils.PointCircleButton("Mode");
    if (ImGui::BeginPopupContextItem("gui_param_mode_button_context", 0)) { // 0 = left mouse button
        if (ImGui::MenuItem("Basic###gui_basic_mode", nullptr, !wc.param_extended_mode, true)) {
            wc.param_extended_mode = false;
        }
        if (ImGui::MenuItem("Expert###gui_expert_mode", nullptr, wc.param_extended_mode, true)) {
            wc.param_extended_mode = true;
        }
        ImGui::EndPopup();
    }
    ImGui::EndGroup();
    /*
    std::string mode_help = "Expert mode enables buttons for additional parameter presentation options.";
    this->utils.HelpMarkerToolTip(mode_help);
    */
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

    /* DISBALED --- Does anybody use this?
    // Toggel Hotkeys
    ImGui::SameLine();
    bool show_only_hotkeys = wc.param_show_hotkeys;
    ImGui::Checkbox("Show Hotkeys", &show_only_hotkeys);
    wc.param_show_hotkeys = show_only_hotkeys;
    */

    // Info
    std::string help_marker = "[INFO]";
    std::string param_help = "[Hover] Show Parameter Description Tooltip\n"
                             "[Right-Click] Context Menu\n"
                             "[Drag & Drop] Move Module to other Parameter Window\n"
                             "[Enter],[Tab],[Left-Click outside Widget] Confirm input changes";
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled(help_marker.c_str());
    this->utils.HoverToolTip(param_help);

    // Paramter substring name filtering (only for main parameter view)
    if (wc.win_callback == WindowManager::DrawCallbacks::MAIN_PARAMETERS) {
        if (std::get<1>(this->hotkeys[GUIWindows::GuiHotkeyIndex::PARAMETER_SEARCH])) {
            this->utils.SetSearchFocus(true);
            std::get<1>(this->hotkeys[GUIWindows::GuiHotkeyIndex::PARAMETER_SEARCH]) = false;
        }
        std::string help_test =
            "[" + std::get<0>(this->hotkeys[GUIWindows::GuiHotkeyIndex::PARAMETER_SEARCH]).ToString() +
            "] Set keyboard focus to search input field.\n"
            "Case insensitive substring search in\nparameter names.\nGlobally in all parameter views.\n";
        this->utils.StringSearch("guiwindow_parameter_earch", help_test);
    }

    /* DISABLED --- Does anybody use this?
    // Module filtering (only for main parameter view)
    if ((this->core_instance != nullptr) && (wc.win_callback == WindowManager::DrawCallbacks::MAIN_PARAMETERS)) {
        std::map<int, std::string> opts;
        opts[static_cast<int>(WindowManager::FilterModes::ALL)] = "All";
        opts[static_cast<int>(WindowManager::FilterModes::INSTANCE)] = "Instance";
        opts[static_cast<int>(WindowManager::FilterModes::VIEW)] = "View";
        unsigned int opts_cnt = (unsigned int)opts.size();
        if (ImGui::BeginCombo("Filter Modules", opts[(int)wc.param_module_filter].c_str())) {
            for (unsigned int i = 0; i < opts_cnt; ++i) {
                if (ImGui::Selectable(opts[i].c_str(), (static_cast<int>(wc.param_module_filter) == i))) {
                    wc.param_module_filter = static_cast<WindowManager::FilterModes>(i);
                    wc.param_modules_list.clear();
                    if ((wc.param_module_filter == WindowManager::FilterModes::INSTANCE) ||
                        (wc.param_module_filter == WindowManager::FilterModes::VIEW)) {
                        std::string viewname;
                        // The goal is to find view module with shortest call connection path to this module.
                        // Since enumeration of modules goes bottom up, result for first abstract view is
                        // stored and following hits are ignored.
                        if (!this->parent_module_fullname.empty()) {
                            const auto view_func = [&, this](core::Module* viewmod) {
                                auto v = dynamic_cast<core::view::AbstractView*>(viewmod);
                                if (v != nullptr) {
                                    std::string vname = v->FullName().PeekBuffer();
                                    bool found = false;
                                    const auto find_func = [&, this](core::Module* guimod) {
                                        std::string modname = guimod->FullName().PeekBuffer();
                                        if (this->parent_module_fullname == modname) {
                                            found = true;
                                        }
                                    };
                                    this->core_instance->EnumModulesNoLock(viewmod, find_func);
                                    if (found && viewname.empty()) {
                                        viewname = vname;
                                    }
                                }
                            };
                            this->core_instance->EnumModulesNoLock(nullptr, view_func);
                        }
                        if (!viewname.empty()) {
                            if (wc.param_module_filter == WindowManager::FilterModes::INSTANCE) {
                                // Considering modules depending on the INSTANCE NAME of the first view this module is
                                // connected to.
                                std::string instname = "";
                                auto instance_idx = viewname.rfind("::");
                                if (instance_idx != std::string::npos) {
                                    instname = viewname.substr(0, instance_idx + 2);
                                }
                                if (!instname.empty()) { // Consider all modules if view is not assigned to any instance
                                    const auto func = [&, this](core::Module* mod) {
                                        std::string modname = mod->FullName().PeekBuffer();
                                        bool foundInstanceName = (modname.find(instname) != std::string::npos);
                                        bool noInstanceNamePresent =
                                            false; /// Always consider modules with no namspace (modname.find("::", 2)
                                                   /// == std::string::npos);
                                        if (foundInstanceName || noInstanceNamePresent) {
                                            wc.param_modules_list.emplace_back(modname);
                                        }
                                    };
                                    this->core_instance->EnumModulesNoLock(nullptr, func);
                                }
                            } else { // (wc.param_module_filter == WindowManager::FilterModes::VIEW)
                                // Considering modules depending on their connection to the VIEW MODULE this GUI is
                                // incorporated.
                                const auto add_func = [&, this](core::Module* mod) {
                                    std::string modname = mod->FullName().PeekBuffer();
                                    wc.param_modules_list.emplace_back(modname);
                                };
                                this->core_instance->EnumModulesNoLock(viewname, add_func);
                            }
                        } else {
                            vislib::sys::Log::DefaultLog.WriteWarn(
                                "Could not find abstract view "
                                "module this gui is connected to. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                        }
                    }
                }
                std::string hover = "Show all Modules."; // == WindowManager::FilterModes::ALL
                if (i == static_cast<int>(WindowManager::FilterModes::INSTANCE)) {
                    hover = "Show Modules with same Instance Name as current View and Modules with no Instance Name.";
                } else if (i == static_cast<int>(WindowManager::FilterModes::VIEW)) {
                    hover = "Show Modules subsequently connected to the View Module the Gui Module is connected to.";
                }
                this->utils.HoverToolTip(hover);
            }
            ImGui::EndCombo();
        }
        this->utils.HelpMarkerToolTip("Selected filter is not refreshed on graph changes.\n"
                                      "Select filter again to trigger refresh.");
    }
    */
    ImGui::Separator();

    // Create child window for sepearte scroll bar and keeping header always visible on top of parameter list
    ImGui::BeginChild("###ParameterList", ImVec2(0.0f, 0.0f), false, ImGuiWindowFlags_HorizontalScrollbar);

    // Listing modules and their parameters
    const core::Module* current_mod = nullptr;
    bool current_mod_open = false;
    const size_t dnd_size = 2048; // Set same max size of all module labels for drag and drop.
    std::string param_namespace = "";
    unsigned int param_indent_stack = 0;
    bool param_namespace_open = true;
    auto currentSearchString = this->utils.GetSearchString();


    configurator::GraphPtrType graph_ptr;
    if (this->graph_manager.GetGraph(this->graph_uid, graph_ptr)) {
        for (auto& module_ptr : graph_ptr->GetModules()) {
            for (auto& param_ptr : module_ptr->parameters) {
            }
        }
    }


    this->core_instance->EnumParameters([&, this](const auto& mod, auto& slot) {
        // Check for new module
        if (current_mod != &mod) {
            current_mod = &mod;
            std::string label = mod.FullName().PeekBuffer();

            // Vertical spacing
            /// if (current_mod_open) ImGui::Dummy(ImVec2(1.0f, ImGui::GetFrameHeightWithSpacing()));

            // Check if module should be considered.
            if (!this->considerModule(label, wc.param_modules_list)) {
                current_mod_open = false;
                return;
            }

            // Reset parameter indent
            param_namespace = "";
            param_namespace_open = true;
            while (param_indent_stack > 0) {
                param_indent_stack--;
                ImGui::Unindent();
            }

            // Determine header state and change color depending on active parameter search
            auto headerId = ImGui::GetID(label.c_str());
            auto headerState = overrideState;
            if (headerState == GUI_INVALID_ID) {
                headerState = ImGui::GetStateStorage()->GetInt(headerId, 0); // 0=close 1=open
            }
            if (!currentSearchString.empty()) {
                headerState = 1;
                ImGui::PushStyleColor(ImGuiCol_Header, ImGui::GetStyleColorVec4(ImGuiCol_PopupBg));
            }
            ImGui::GetStateStorage()->SetInt(headerId, headerState);

            current_mod_open = ImGui::CollapsingHeader(label.c_str(), nullptr);

            if (!currentSearchString.empty()) {
                ImGui::PopStyleColor();
            }

            // Set parameter indent
            param_indent_stack++;
            ImGui::Indent();

            // Module description as hover tooltip
            auto mod_desc = this->core_instance->GetModuleDescriptionManager().Find(mod.ClassName());
            if (mod_desc != nullptr) {
                this->utils.HoverToolTip(std::string(mod_desc->Description()), ImGui::GetID(label.c_str()), 0.5f, 5.0f);
            }

            // Context menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Copy to new Window")) {
                    // using instance time as hidden unique id
                    std::string window_name =
                        "Parameters###parameters_" + std::to_string(this->state.last_instance_time);
                    WindowManager::WindowConfiguration buf_win;
                    buf_win.win_name = window_name;
                    buf_win.win_show = true;
                    buf_win.win_flags = ImGuiWindowFlags_HorizontalScrollbar;
                    buf_win.win_callback = WindowManager::DrawCallbacks::PARAMETERS;
                    buf_win.param_show_hotkeys = false;
                    buf_win.win_position = ImVec2(0.0f, ImGui::GetTextLineHeightWithSpacing());
                    buf_win.win_size = ImVec2(400.0f, 600.0f);
                    buf_win.param_modules_list.emplace_back(label);
                    this->window_manager.AddWindowConfiguration(buf_win);
                }

                // Deleting module's parameters is not available in main parameter window.
                if (wc.win_callback != WindowManager::DrawCallbacks::MAIN_PARAMETERS) {
                    if (ImGui::MenuItem("Delete from List")) {
                        std::vector<std::string>::iterator find_iter =
                            std::find(wc.param_modules_list.begin(), wc.param_modules_list.end(), label);
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
            label.resize(dnd_size);
            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
                ImGui::SetDragDropPayload("DND_COPY_MODULE_PARAMETERS", label.c_str(), (label.size() * sizeof(char)));
                ImGui::TextUnformatted(label.c_str());
                ImGui::EndDragDropSource();
            }
        }

        if (current_mod_open) {
            auto parameter_ptr = slot.Parameter();
            std::string param_name = slot.Name().PeekBuffer();
            bool showSearchedParameter = true;
            if (!currentSearchString.empty()) {
                showSearchedParameter = this->utils.FindCaseInsensitiveSubstring(param_name, currentSearchString);
            }

            bool param_visible = ((parameter_ptr->IsGUIVisible() || wc.param_extended_mode) && showSearchedParameter);
            if (!parameter_ptr.IsNull() && param_visible) {

                // Parameter namespace header
                auto pos = param_name.find("::");
                std::string current_param_namespace = "";
                if (pos != std::string::npos) {
                    current_param_namespace = param_name.substr(0, pos);
                }
                if (current_param_namespace != param_namespace) {
                    param_namespace = current_param_namespace;
                    while (param_indent_stack > 1) {
                        param_indent_stack--;
                        ImGui::Unindent();
                    }
                    /// ImGui::Separator();
                    if (!param_namespace.empty()) {

                        std::string label = param_namespace + "###" + param_namespace + "__" + param_name;
                        if (!currentSearchString.empty()) {
                            auto headerId = ImGui::GetID(label.c_str());
                            ImGui::GetStateStorage()->SetInt(headerId, 1);
                        }
                        param_namespace_open = ImGui::CollapsingHeader(label.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
                        param_indent_stack++;
                        ImGui::Indent();
                    } else {
                        param_namespace_open = true;
                    }
                }

                // Set general proportional parameter item width
                float widget_width = ImGui::GetContentRegionAvail().x * 0.6f;
                /// widget_width -= (static_cast<float>(param_indent_stack) * style.IndentSpacing);
                ImGui::PushItemWidth(widget_width);

                // Draw parameter
                if (param_namespace_open) {
                    /* DISABLED
                    if (wc.param_show_hotkeys) {
                        if (auto* p = slot.template Param<core::param::ButtonParam>()) {
                            std::string label = slot.Name().PeekBuffer();
                            std::string desc = slot.Description().PeekBuffer();
                            std::string keycode = p->GetKeyCode().ToString();
                            ImGui::Columns(2, "hotkey_columns", false);
                            ImGui::TextUnformatted(label.c_str());
                            this->utils.HoverToolTip(desc);
                            ImGui::NextColumn();
                            ImGui::TextUnformatted(keycode.c_str());
                            this->utils.HoverToolTip(desc);
                            // Reset colums
                            ImGui::Columns(1);
                            ImGui::Separator();
                        }
                    } else {
                    */
                    auto param_ref = &(*parameter_ptr);
                    auto param_ptr = this->param_core_interface_map[param_ref];
                    this->drawParameter(
                        param_ptr, configurator::ParameterPresentation::WidgetScope::LOCAL, wc.param_extended_mode);
                    /*
                    }
                    */
                }

                ImGui::PopItemWidth();
            }
        }
    });


    // Reset parameter namespace stuff
    while (param_indent_stack > 0) {
        param_indent_stack--;
        ImGui::Unindent();
    }

    // Drop target
    ImGui::Dummy(ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetFontSize()));
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_COPY_MODULE_PARAMETERS")) {

            IM_ASSERT(payload->DataSize == (dnd_size * sizeof(char)));
            std::string payload_id = (const char*)payload->Data;

            // Insert dragged module name only if not contained in list
            if (!this->considerModule(payload_id, wc.param_modules_list)) {
                wc.param_modules_list.emplace_back(payload_id);
            }
        }
        ImGui::EndDragDropTarget();
    }

    ImGui::EndChild();
}


void GUIWindows::drawFpsWindowCallback(WindowManager::WindowConfiguration& wc) {
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
    if (ImGui::RadioButton("fps", (wc.ms_mode == WindowManager::TimingModes::FPS))) {
        wc.ms_mode = WindowManager::TimingModes::FPS;
    }
    ImGui::SameLine();

    if (ImGui::RadioButton("ms", (wc.ms_mode == WindowManager::TimingModes::MS))) {
        wc.ms_mode = WindowManager::TimingModes::MS;
    }

    ImGui::TextDisabled("Frame ID:");
    ImGui::SameLine();
    ImGui::Text("%u", this->core_instance->GetFrameID());

    ImGui::SameLine(
        ImGui::CalcItemWidth() - (ImGui::GetFrameHeightWithSpacing() - style.ItemSpacing.x - style.ItemInnerSpacing.x));
    if (ImGui::ArrowButton("Options_", ((wc.ms_show_options) ? (ImGuiDir_Down) : (ImGuiDir_Up)))) {
        wc.ms_show_options = !wc.ms_show_options;
    }

    std::vector<float> value_array = wc.buf_values;
    if (wc.ms_mode == WindowManager::TimingModes::FPS) {
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
    if (wc.ms_mode == WindowManager::TimingModes::FPS) {
        plot_scale_factor *= wc.buf_plot_fps_scaling;
    } else if (wc.ms_mode == WindowManager::TimingModes::MS) {
        plot_scale_factor *= wc.buf_plot_ms_scaling;
    }

    ImGui::PlotLines(
        "###msplot", value_ptr, buffer_size, 0, overlay.c_str(), 0.0f, plot_scale_factor, ImVec2(0.0f, 50.0f));

    if (wc.ms_show_options) {
        if (ImGui::InputFloat(
                "Refresh Rate", &wc.ms_refresh_rate, 1.0f, 10.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue)) {
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
            vislib::sys::Log::DefaultLog.WriteWarn(
                "No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            vislib::sys::Log::DefaultLog.WriteInfo("[GUI] Current Performance Monitor Value:\n%s", overlay.c_str());
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
            vislib::sys::Log::DefaultLog.WriteWarn(
                "No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            vislib::sys::Log::DefaultLog.WriteInfo("[GUI] All Performance Monitor Values:\n%s", stream.str().c_str());
#endif
        }
        ImGui::SameLine();
        ImGui::TextUnformatted("Copy to Clipborad");
        std::string help = "Values are copied in chronological order (newest first)";
        this->utils.HelpMarkerToolTip(help);
    }
}


void GUIWindows::drawFontWindowCallback(WindowManager::WindowConfiguration& wc) {
    ImGuiIO& io = ImGui::GetIO();

    ImFont* font_current = ImGui::GetFont();
    if (ImGui::BeginCombo("Select available Font", font_current->GetDebugName())) {
        for (int n = 0; n < (io.Fonts->Fonts.Size - 1); n++) { // ! n < size-1 for skipping last added font which is
                                                               // exclusively used by configurator for the graph.
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
    std::string help = "Same font can be loaded multiple times with different font size.";
    this->utils.HelpMarkerToolTip(help);

    std::string label = "Font Size";
    ImGui::InputFloat(label.c_str(), &wc.buf_font_size, 1.0f, 10.0f, "%.2f", ImGuiInputTextFlags_None);
    // Validate font size
    if (wc.buf_font_size <= 0.0f) {
        wc.buf_font_size = 5.0f; // minimum valid font size
    }

    label = "Font File Name (.ttf)";
    /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
    GUIUtils::Utf8Encode(wc.buf_font_file);
    ImGui::InputText(label.c_str(), &wc.buf_font_file, ImGuiInputTextFlags_AutoSelectAll);
    GUIUtils::Utf8Decode(wc.buf_font_file);
    // Validate font file before offering load button
    if (FileUtils::FilesExistingExtension<std::string>(wc.buf_font_file, std::string(".ttf"))) {
        if (ImGui::Button("Add Font")) {
            this->state.font_file = wc.buf_font_file;
            this->state.font_size = wc.buf_font_size;
        }
    } else {
        ImGui::TextColored(GUI_COLOR_TEXT_ERROR, "Please enter valid font file name.");
    }
}


void GUIWindows::drawParameter(std::shared_ptr<megamol::gui::configurator::Parameter> param_ptr,
    megamol::gui::configurator::ParameterPresentation::WidgetScope scope, bool extended) {

    if (param_ptr == nullptr) return;

    param_ptr->present.expert = extended;

    if (param_ptr->type == ParamType::TRANSFERFUNCTION) {
        param_ptr->present.ConnectExternalTransferFunctionEditor(this->tf_editor_ptr);
    }

    if (param_ptr->PresentGUI(scope)) {
        if (scope == megamol::gui::configurator::ParameterPresentation::WidgetScope::LOCAL) {

            // Open window calling the transfer function editor callback
            if ((param_ptr->type == ParamType::TRANSFERFUNCTION)) {
                this->tf_editor_ptr->SetConnectedParameter(param_ptr);
                const auto func = [](WindowManager::WindowConfiguration& wc) {
                    if (wc.win_callback == WindowManager::DrawCallbacks::TRANSFER_FUNCTION) {
                        wc.win_show = true;
                    }
                };
                this->window_manager.EnumWindows(func);
            }
        }
    }
}


void GUIWindows::drawMenu(void) {

    if (ImGui::BeginMenu("File")) {
        // Load/save parameter values to LUA file
        if (ImGui::MenuItem("Save Running Project",
                std::get<0>(this->hotkeys[GUIWindows::GuiHotkeyIndex::SAVE_PROJECT]).ToString().c_str())) {
            this->state.open_popup_save = true;
        }

        if (ImGui::MenuItem("Exit", "ALT + 'F4'")) {
            // Exit program
            this->shutdown();
        }
        ImGui::EndMenu();
    }

    // Windows
    if (ImGui::BeginMenu("Windows")) {

        if (ImGui::MenuItem("Menu", std::get<0>(this->hotkeys[GUIWindows::GuiHotkeyIndex::MENU]).ToString().c_str(),
                &this->state.menu_visible)) {
            this->state.menu_visible = !this->state.menu_visible;
        }

        const auto func = [&, this](WindowManager::WindowConfiguration& wc) {
            bool win_open = wc.win_show;
            std::string hotkey_label = wc.win_hotkey.ToString();
            if (!hotkey_label.empty()) {
                hotkey_label = "(SHIFT +) " + hotkey_label;
            }
            if (ImGui::MenuItem(wc.win_name.c_str(), hotkey_label.c_str(), &win_open)) {
                wc.win_show = !wc.win_show;
            }
            // Add conext menu for deleting windows without hotkey (= custom parameter windows).
            if (wc.win_hotkey.key == core::view::Key::KEY_UNKNOWN) {
                if (ImGui::BeginPopupContextItem()) {
                    if (ImGui::MenuItem("Delete Window")) {
                        this->state.win_delete = wc.win_name;
                    }
                    ImGui::EndPopup();
                }
                this->utils.HoverToolTip("[Right-Click] Open Context Menu for Deleting Window Permanently.");
            } else {
                this->utils.HoverToolTip("['Window Hotkey'] Show/Hide Window.\n[Shift]+['Window Hotkey'] Reset Size "
                                         "and Position of Window.");
            }
        };
        this->window_manager.EnumWindows(func);

        ImGui::EndMenu();
    }

    // Help
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

        const std::string eMail = "megamol@visus.uni-stuttgart.de";
        const std::string webLink = "https://megamol.org/";
        const std::string gitLink = "https://github.com/UniStuttgart-VISUS/megamol";

        std::string about = std::string("MegaMol - Version ") + std::to_string(MEGAMOL_CORE_MAJOR_VER) + (".") +
                            std::to_string(MEGAMOL_CORE_MINOR_VER) + ("\ngit# ") + std::string(MEGAMOL_CORE_COMP_REV) +
                            ("\nDear ImGui - Version ") + std::string(IMGUI_VERSION) + ("\n");
        std::string mailstr = std::string("Contact: ") + eMail;
        std::string webstr = std::string("Web: ") + webLink;
        std::string gitstr = std::string("Git-Hub: ") + gitLink;

        ImGui::TextUnformatted(about.c_str());

        ImGui::Separator();
        if (ImGui::Button("Copy E-Mail")) {
#ifdef GUI_USE_GLFW
            auto glfw_win = ::glfwGetCurrentContext();
            ::glfwSetClipboardString(glfw_win, eMail.c_str());
#elif _WIN32
            ImGui::SetClipboardText(eMail.c_str());
#else // LINUX
            vislib::sys::Log::DefaultLog.WriteWarn(
                "No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            vislib::sys::Log::DefaultLog.WriteInfo("[GUI] E-Mail address:\n%s", eMail.c_str());
#endif
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(mailstr.c_str());


        if (ImGui::Button("Copy Website")) {
#ifdef GUI_USE_GLFW
            auto glfw_win = ::glfwGetCurrentContext();
            ::glfwSetClipboardString(glfw_win, webLink.c_str());
#elif _WIN32
            ImGui::SetClipboardText(webLink.c_str());
#else // LINUX
            vislib::sys::Log::DefaultLog.WriteWarn(
                "No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            vislib::sys::Log::DefaultLog.WriteInfo("[GUI] Website link:\n%s", webLink.c_str());
#endif
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(webstr.c_str());

        if (ImGui::Button("Copy GitHub")) {
#ifdef GUI_USE_GLFW
            auto glfw_win = ::glfwGetCurrentContext();
            ::glfwSetClipboardString(glfw_win, gitLink.c_str());
#elif _WIN32
            ImGui::SetClipboardText(gitLink.c_str());
#else // LINUX
            vislib::sys::Log::DefaultLog.WriteWarn(
                "No clipboard use provided. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            vislib::sys::Log::DefaultLog.WriteInfo("[GUI] GitHub link:\n%s", gitLink.c_str());
#endif
        }
        ImGui::SameLine();
        ImGui::TextUnformatted(gitstr.c_str());

        ImGui::Separator();
        about = "Copyright (C) 2009-2019 by Universitaet Stuttgart "
                "(VIS).\nAll rights reserved.";
        ImGui::TextUnformatted(about.c_str());

        ImGui::Separator();
        if (ImGui::Button("Close")) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    // Save project pop-up
    this->state.open_popup_save =
        (this->state.open_popup_save || std::get<1>(this->hotkeys[GUIWindows::GuiHotkeyIndex::SAVE_PROJECT]));

    bool confirmed, aborted;
    bool popup_failed = false;
    std::string project_filename;
    configurator::GraphPtrType graph_ptr;
    if (this->graph_manager.GetGraph(this->graph_uid, graph_ptr)) {
        project_filename = graph_ptr->GetFilename();
    }
    if (this->file_utils.FileBrowserPopUp(
            FileUtils::FileBrowserFlag::SAVE, "Save Editor Project", this->state.open_popup_save, project_filename)) {
        // Serialize current state to parameter.
        this->save_state_to_parameter();
        // Save to file
        popup_failed = !this->graph_manager.SaveProjectToFile(this->graph_uid, project_filename);
    }
    this->utils.MinimalPopUp("Failed to Save Project", popup_failed, "See console log output for more information.", "",
        confirmed, "Cancel", aborted);

    if (this->state.open_popup_save) {
        this->state.open_popup_save = false;
        std::get<1>(this->hotkeys[GUIWindows::GuiHotkeyIndex::SAVE_PROJECT]) = false;
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
            hotkeylist.emplace_back(std::get<0>(h));
        }

        // Add hotkeys of configurator
        for (auto& h : this->configurator.GetHotkeys()) {
            hotkeylist.emplace_back(std::get<0>(h));
        }

        for (auto& pair : this->param_core_interface_map) {
            auto param_ptr = pair.second;
            if (param_ptr == nullptr) continue;
            if (param_ptr->type == ParamType::BUTTON) {
                auto keyCode = param_ptr->GetStorage<megamol::core::view::KeyCode>();
                // Ignore not set hotekey
                if (keyCode.key == core::view::Key::KEY_UNKNOWN) {
                    return;
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
                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "The hotkey [%s] of the parameter \"%s\" has already been assigned. "
                        ">>> If this hotkey is pressed, there will be no effect on this parameter!",
                        keyCode.ToString().c_str(), param_ptr->full_name.c_str());
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


void megamol::gui::GUIWindows::shutdown(void) {

    if (this->core_instance != nullptr) {
#ifdef GUI_VERBOSE
        vislib::sys::Log::DefaultLog.WriteInfo("[GUI] Shutdown MegaMol instance.");
#endif // GUI_VERBOSE
        this->core_instance->Shutdown();
    } else {
        vislib::sys::Log::DefaultLog.WriteError(
            "Pointer to core instance is nullptr. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
    }
}


void megamol::gui::GUIWindows::save_state_to_parameter(void) {

    this->configurator.UpdateStateParameter();

    nlohmann::json window_json;
    nlohmann::json gui_parameter_json;

    if (this->window_manager.StateToJSON(window_json) && this->gui_and_parameters_state_to_json(gui_parameter_json)) {
        // Merge all JSON states
        window_json.update(gui_parameter_json);

        std::string state;
        state = window_json.dump(2);
        this->state_param.Param<core::param::StringParam>()->SetValue(state.c_str(), false);

        auto parameter_slot_ptr = this->state_param.Parameter();
        if (parameter_slot_ptr.IsNull()) {
            return;
        }
        auto param_ref = &(*parameter_slot_ptr);
        auto param_ptr = this->param_core_interface_map[param_ref];
        if (param_ptr == nullptr) return;
        megamol::gui::configurator::ReadCoreParameter(this->state_param, (*param_ptr), false);
    }
}


bool megamol::gui::GUIWindows::gui_and_parameters_state_from_json_string(const std::string& in_json_string) {

    try {
        if (in_json_string.empty()) {
            return false;
        }

        bool found_parameters = false;
        bool found_gui = false;
        bool valid = true;
        nlohmann::json json;
        json = nlohmann::json::parse(in_json_string);
        if (!json.is_object()) {
            vislib::sys::Log::DefaultLog.WriteError(
                "State is no valid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        for (auto& header_item : json.items()) {
            if (header_item.key() == GUI_JSON_TAG_GUISTATE) {
                found_gui = true;
                auto gui_state = header_item.value();

                // menu_visible
                if (gui_state.at("menu_visible").is_boolean()) {
                    gui_state.at("menu_visible").get_to(this->state.menu_visible);
                } else {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "JSON state: Failed to read 'menu_visible' as boolean. [%s, %s, line %d]\n", __FILE__,
                        __FUNCTION__, __LINE__);
                }
                // project_file (supports UTF-8)
                if (gui_state.at("project_file").is_string()) {
                    gui_state.at("project_file").get_to(this->state.project_file);
                    this->utils.Utf8Decode(this->state.project_file);
                } else {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "JSON state: Failed to read 'project_file' as string. [%s, %s, line %d]\n", __FILE__,
                        __FUNCTION__, __LINE__);
                }
            } else if (header_item.key() == GUI_JSON_TAG_GUISTATE_PARAMETERS) {
                /// XXX ! Implementation should be duplicate to Configurator-Version
                /// XXX megamol::gui::configurator::GraphManager::parameters_gui_state_from_json_string()

                found_parameters = true;
                for (auto& config_item : header_item.value().items()) {
                    std::string json_param_name = config_item.key();
                    auto gui_state = config_item.value();
                    valid = true;

                    // gui_visibility
                    bool gui_visibility;
                    if (gui_state.at("gui_visibility").is_boolean()) {
                        gui_state.at("gui_visibility").get_to(gui_visibility);
                    } else {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "JSON state: Failed to read 'gui_visibility' as boolean. [%s, %s, line %d]\n", __FILE__,
                            __FUNCTION__, __LINE__);
                        valid = false;
                    }

                    // gui_read-only
                    bool gui_read_only;
                    if (gui_state.at("gui_read-only").is_boolean()) {
                        gui_state.at("gui_read-only").get_to(gui_read_only);
                    } else {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "JSON state: Failed to read 'gui_read-only' as boolean. [%s, %s, line %d]\n", __FILE__,
                            __FUNCTION__, __LINE__);
                        valid = false;
                    }

                    // gui_presentation_mode
                    PresentType gui_presentation_mode;
                    if (gui_state.at("gui_presentation_mode").is_number_integer()) {
                        gui_presentation_mode =
                            static_cast<PresentType>(gui_state.at("gui_presentation_mode").get<int>());
                    } else {
                        vislib::sys::Log::DefaultLog.WriteError(
                            "JSON state: Failed to read 'gui_presentation_mode' as integer. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }

                    if (valid) {
                        for (auto& pair : this->param_core_interface_map) {
                            auto param_ptr = pair.second;
                            if (param_ptr != nullptr) {
                                if (json_param_name == param_ptr->full_name) {
                                    param_ptr->present.SetGUIVisible(gui_visibility);
                                    param_ptr->present.SetGUIReadOnly(gui_read_only);
                                    param_ptr->present.SetGUIPresentation(gui_presentation_mode);
                                    /// param_ptr->ForceSetDirty();
                                }
                            }
                        }
                    }
                }
            }
        }

        if (found_parameters && found_gui) {
#ifdef GUI_VERBOSE
            vislib::sys::Log::DefaultLog.WriteInfo("[GUI] Read gui and parameter state from JSON string.");
#endif // GUI_VERBOSE
        } else {
            /// vislib::sys::Log::DefaultLog.WriteWarn("Could not find gui or parameter state in JSON. [%s, %s, line
            /// %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

    } catch (nlohmann::json::type_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::invalid_iterator& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::out_of_range& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::other_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Unknown Error - Unable to parse JSON string. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


bool megamol::gui::GUIWindows::gui_and_parameters_state_to_json(nlohmann::json& out_json) {

    try {
        /// Append to given json
        // out_json.clear();

        this->utils.Utf8Encode(this->state.project_file);
        out_json[GUI_JSON_TAG_GUISTATE]["project_file"] = this->state.project_file;
        out_json[GUI_JSON_TAG_GUISTATE]["menu_visible"] = this->state.menu_visible;

        /// XXX ! Implementation should be duplicate to Configurator-Version
        /// XXX megamol::gui::configurator::GraphManager::parameters_gui_state_to_json()
        for (auto& pair : this->param_core_interface_map) {
            auto param_ptr = pair.second;
            if (param_ptr != nullptr) {
                std::string param_name = param_ptr->full_name;
                out_json[GUI_JSON_TAG_GUISTATE_PARAMETERS][param_name]["gui_visibility"] =
                    param_ptr->present.IsGUIVisible();
                out_json[GUI_JSON_TAG_GUISTATE_PARAMETERS][param_name]["gui_read-only"] =
                    param_ptr->present.IsGUIReadOnly();
                out_json[GUI_JSON_TAG_GUISTATE_PARAMETERS][param_name]["gui_presentation_mode"] =
                    static_cast<int>(param_ptr->present.GetGUIPresentation());
            }
        }

#ifdef GUI_VERBOSE
        vislib::sys::Log::DefaultLog.WriteInfo("[GUI] Wrote parameter state to JSON.");
#endif // GUI_VERBOSE

    } catch (nlohmann::json::type_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::invalid_iterator& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::out_of_range& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::other_error& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Unknown Error - Unable to write JSON of state. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}
