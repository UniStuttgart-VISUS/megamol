/*
 * GUIView.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

/**
 * USED HOTKEYS:
 *
 * - Show/hide Windows: Ctrl + F9-F12
 * - Reset windows:     Shift + (Window show/hide hotkeys)
 * - Quit program:      Alt + F4
 */

#include "stdafx.h"
#include "GUIView.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/Module.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/TernaryParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/versioninfo.h"
#include "mmcore/view/CallRenderView.h"

#include <imgui_internal.h>
#include "CorporateGreyStyle.h"
#include "CorporateWhiteStyle.h"
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"

#include <algorithm>
#include <iomanip>
#include <sstream>


#define GUI_MAX_MULITLINE 7


using namespace megamol;
using namespace megamol::gui;
using vislib::sys::Log;


GUIView::GUIView()
    : core::view::AbstractView()
    , render_view_slot("renderview", "Connects to a preceding RenderView that will be decorated with a GUI")
    , style_param("style", "Color style, theme")
    , state_param("state", "Current state of all windows. Automatically updated.")
    , context(nullptr)
    , window_manager()
    , tf_editor()
    , utils()
    , state()
    , widgtmap_text()
    , widgtmap_float()
    , widgtmap_int()
    , widgtmap_vec2()
    , widgtmap_vec3()
    , widgtmap_vec4() {

    this->render_view_slot.SetCompatibleCall<core::view::CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->render_view_slot);

    core::param::EnumParam* styles = new core::param::EnumParam((int)(Styles::DarkColors));
    styles->SetTypePair(Styles::CorporateGray, "Corporate Gray");
    styles->SetTypePair(Styles::CorporateWhite, "Corporate White");
    styles->SetTypePair(Styles::DarkColors, "Dark Colors");
    styles->SetTypePair(Styles::LightColors, "Light Colors");
    this->style_param << styles;
    this->style_param.ForceSetDirty();
    this->MakeSlotAvailable(&this->style_param);

    this->state_param << new core::param::StringParam("");
    this->MakeSlotAvailable(&this->state_param);
}

GUIView::~GUIView() { this->Release(); }

bool GUIView::create() {
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
        vislib::sys::Log::DefaultLog.WriteError("[GUIView] Could not create ImGui context");
        return false;
    }
    ImGui::SetCurrentContext(this->context);

    // Init OpenGL for ImGui --------------------------------------------------
    const char* glsl_version = "#version 130"; /// "#version 150"
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Register window callbacks in window manager ----------------------------
    this->window_manager.RegisterDrawWindowCallback(
        WindowManager::DrawCallbacks::MAIN, [&, this](const std::string& wn, WindowManager::WindowConfiguration& wc) {
            this->drawMainWindowCallback(wn, wc);
        });
    this->window_manager.RegisterDrawWindowCallback(
        WindowManager::DrawCallbacks::PARAM, [&, this](const std::string& wn, WindowManager::WindowConfiguration& wc) {
            this->drawParametersCallback(wn, wc);
        });
    this->window_manager.RegisterDrawWindowCallback(
        WindowManager::DrawCallbacks::FPSMS, [&, this](const std::string& wn, WindowManager::WindowConfiguration& wc) {
            this->drawFpsWindowCallback(wn, wc);
        });
    this->window_manager.RegisterDrawWindowCallback(
        WindowManager::DrawCallbacks::FONT, [&, this](const std::string& wn, WindowManager::WindowConfiguration& wc) {
            this->drawFontWindowCallback(wn, wc);
        });
    this->window_manager.RegisterDrawWindowCallback(
        WindowManager::DrawCallbacks::TF, [&, this](const std::string& wn, WindowManager::WindowConfiguration& wc) {
            this->drawTFWindowCallback(wn, wc);
        });

    // Create window configurations
    WindowManager::WindowConfiguration buf_win;
    buf_win.buf_win_reset = true;
    // MAIN Window ------------------------------------------------------------
    buf_win.win_show = true;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F12, core::view::Modifier::CTRL);
    buf_win.win_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoTitleBar;
    buf_win.win_callback = WindowManager::DrawCallbacks::MAIN;
    buf_win.win_position = ImVec2(12, 12);
    buf_win.win_size = ImVec2(250, 600);
    this->window_manager.AddWindowConfiguration("Main Window", buf_win);

    // FPS/MS Window ----------------------------------------------------------
    buf_win.win_show = false;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F11, core::view::Modifier::CTRL);
    buf_win.win_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
    buf_win.win_callback = WindowManager::DrawCallbacks::FPSMS;
    this->window_manager.AddWindowConfiguration("Performance Metrics", buf_win);

    // FONT Window ------------------------------------------------------------
    buf_win.win_show = false;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F10, core::view::Modifier::CTRL);
    buf_win.win_flags = ImGuiWindowFlags_AlwaysAutoResize;
    buf_win.win_callback = WindowManager::DrawCallbacks::FONT;
    this->window_manager.AddWindowConfiguration("Font Settings", buf_win);

    // TRANSFER FUNCTION Window -----------------------------------------------
    buf_win.win_show = false;
    buf_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F9, core::view::Modifier::CTRL);
    buf_win.win_flags = ImGuiWindowFlags_AlwaysAutoResize;
    buf_win.win_callback = WindowManager::DrawCallbacks::TF;
    this->window_manager.AddWindowConfiguration("Transfer Function Editor", buf_win);

    // Style settings ---------------------------------------------------------
    ImGui::SetColorEditOptions(ImGuiColorEditFlags_Uint8 | ImGuiColorEditFlags_RGB |
                               ImGuiColorEditFlags_PickerHueWheel | ImGuiColorEditFlags_AlphaBar |
                               ImGuiColorEditFlags_AlphaPreview);

    // IO settings ------------------------------------------------------------
    ImGuiIO& io = ImGui::GetIO();
    io.IniSavingRate = 5.0f;          //  in seconds
    io.IniFilename = nullptr;         // "imgui.ini"; - disabled, using own window settings profile
    io.LogFilename = "imgui_log.txt"; // (set to nullptr to disable)
    io.FontAllowUserScaling = true;

    // Init global state -------------------------------------------------------
    this->state.font_file = "";
    this->state.font_size = 13.0f;
    this->state.font_index = -1;
    this->state.win_save_state = false;
    this->state.win_save_delay = 0.0f;
    this->state.win_delete = "";
    this->state.last_instance_time = 0.0f;
    this->state.hotkeys_check_once = true;
    // Adding additional utf-8 glyph ranges
    /// (there is no error if glyph has no representation in font atlas)
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
        ImFontConfig config;
        config.OversampleH = 6;
        config.GlyphRanges = this->state.font_utf8_ranges.data();
        // Add default font
        io.Fonts->AddFontDefault(&config);
#ifdef GUI_USE_FILEUTILS
        // Add other known fonts
        std::string font_file, font_path;
        const vislib::Array<vislib::StringW>& searchPaths =
            this->GetCoreInstance()->Configuration().ResourceDirectories();
        for (size_t i = 0; i < searchPaths.Count(); ++i) {
            std::wstring searchPath(searchPaths[i].PeekBuffer());
            font_file = "Roboto-Regular.ttf";
            font_path = SearchFileRecursive(font_file, searchPath);
            if (!font_path.empty()) {
                io.Fonts->AddFontFromFileTTF(font_path.c_str(), 12.0f, &config);
                // Set as default.
                io.FontDefault = io.Fonts->Fonts[(io.Fonts->Fonts.Size - 1)];
            }
            font_file = "SourceCodePro-Regular.ttf";
            font_path = SearchFileRecursive(font_file, searchPath);
            if (!font_path.empty()) {
                io.Fonts->AddFontFromFileTTF(font_path.c_str(), 13.0f, &config);
            }
        }
#endif // GUI_USE_FILEUTILS
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


void GUIView::release() {
    if (this->context != nullptr) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui::DestroyContext(this->context);
    }
}


void GUIView::unpackMouseCoordinates(float& x, float& y) {
    GLint vpw = 1;
    GLint vph = 1;
    if (this->overrideCall == nullptr) {
        GLint vp[4];
        ::glGetIntegerv(GL_VIEWPORT, vp);
        vpw = vp[2];
        vph = vp[3];
    } else {
        vpw = this->overrideCall->ViewportWidth();
        vph = this->overrideCall->ViewportHeight();
    }
    x *= static_cast<float>(vpw);
    y *= static_cast<float>(vph);
}


float GUIView::DefaultTime(double instTime) const {
    // This view does not do any time control
    return 0.0f;
}


unsigned int GUIView::GetCameraSyncNumber(void) const {
    Log::DefaultLog.WriteWarn("GUIView::GetCameraSyncNumber unsupported");
    return 0u;
}


void GUIView::SerialiseCamera(vislib::Serialiser& serialiser) const {
    Log::DefaultLog.WriteWarn("GUIView::SerialiseCamera unsupported");
}


void GUIView::DeserialiseCamera(vislib::Serialiser& serialiser) {
    Log::DefaultLog.WriteWarn("GUIView::DeserialiseCamera unsupported");
}


void GUIView::Render(const mmcRenderViewContext& context) {
    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
    if (this->doHookCode()) {
        this->doBeforeRenderHook();
    }
    if (crv) {
        crv->SetOutputBuffer(GL_BACK);
        crv->SetInstanceTime(context.InstanceTime);
        crv->SetTime(
            -1.0f); // Should be negative to trigger animation! (see View3D.cpp line ~660 | View2D.cpp line ~350)
        (*crv)(core::view::AbstractCallRender::FnRender);
        this->drawGUI(crv->GetViewport(), crv->InstanceTime());
    } else {
        ::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        ::glClear(GL_COLOR_BUFFER_BIT);
        if (this->overrideCall != nullptr) {
            this->drawGUI(this->overrideCall->GetViewport(), context.InstanceTime);
        }
    }
    if (this->doHookCode()) {
        this->doAfterRenderHook();
    }
}


void GUIView::ResetView(void) {
    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
    if (crv) {
        (*crv)(core::view::CallRenderView::CALL_RESETVIEW);
    }
}


void GUIView::Resize(unsigned int width, unsigned int height) {
    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
    if (crv) {
        // der ganz ganz dicke "because-i-know"-Knueppel
        AbstractView* view = const_cast<AbstractView*>(
            dynamic_cast<const AbstractView*>(static_cast<const Module*>(crv->PeekCalleeSlot()->Owner())));
        if (view != nullptr) {
            view->Resize(width, height);
        }
    }
}


void GUIView::UpdateFreeze(bool freeze) {
    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
    if (crv) {
        auto callType = freeze ? core::view::CallRenderView::CALL_FREEZE : core::view::CallRenderView::CALL_UNFREEZE;
        (*crv)(callType);
    }
}


bool GUIView::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {
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

    // Check for additional text modification hotkeys
    if (action == core::view::KeyAction::RELEASE) {
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_A)] = false;
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_C)] = false;
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_V)] = false;
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_X)] = false;
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_Y)] = false;
        io.KeysDown[static_cast<size_t>(GuiTextModHotkeys::CTRL_Z)] = false;
    }
    bool hotkeyPressed = true;
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
    }

    // ------------------------------------------------------------------------
    // NB: Hotkey processing is stopped after first occurence. Order of hotkey processing is crucial.
    // Hotkeys always trigger just oneevent.

    // Exit megamol
    hotkeyPressed = ((io.KeyAlt) && (ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_F4)))); // Alt + F4
    if (hotkeyPressed) {
        this->shutdown();
        return true;
    }

    // Hotkeys of window(s)
    const auto func = [&, this](const std::string& wn, WindowManager::WindowConfiguration& wc) {
        hotkeyPressed = (ImGui::IsKeyDown(static_cast<int>(wc.win_hotkey.key))) &&
                        (wc.win_hotkey.mods.test(core::view::Modifier::CTRL) == io.KeyCtrl);
        if (hotkeyPressed) {
            if (io.KeyShift) {
                wc.win_soft_reset = true;
            } else {
                wc.win_show = !wc.win_show;
            }
        }
    };
    this->window_manager.EnumWindows(func);

    // Always consume keyboard input if requested by any imgui widget (e.g. text input).
    // User expects hotkey priority of text input thus needs to be processed before parameter hotkeys.
    if (io.WantCaptureKeyboard) {
        return true;
    }

    // Check only considered modules for pressed parameter hotkeys
    std::vector<std::string> modules_list;
    const auto modfunc = [&, this](const std::string& wn, WindowManager::WindowConfiguration& wc) {
        for (auto& m : wc.param_modules_list) {
            modules_list.emplace_back(m);
        }
    };
    this->window_manager.EnumWindows(modfunc);
    hotkeyPressed = false;
    const core::Module* current_mod = nullptr;
    bool consider_module = false;
    this->GetCoreInstance()->EnumParameters([&, this](const auto& mod, auto& slot) {
        if (current_mod != &mod) {
            current_mod = &mod;
            consider_module = this->considerModule(mod.FullName().PeekBuffer(), modules_list);
        }

        if (consider_module) {
            auto param = slot.Parameter();
            if (!param.IsNull()) {
                if (auto* p = slot.template Param<core::param::ButtonParam>()) {
                    auto keyCode = p->GetKeyCode();

                    // Break loop after first occurrence of parameter hotkey
                    if (hotkeyPressed) return;

                    hotkeyPressed = (ImGui::IsKeyDown(static_cast<int>(keyCode.key))) &&
                                    (keyCode.mods.test(core::view::Modifier::ALT) == io.KeyAlt) &&
                                    (keyCode.mods.test(core::view::Modifier::CTRL) == io.KeyCtrl) &&
                                    (keyCode.mods.test(core::view::Modifier::SHIFT) == io.KeyShift);
                    if (hotkeyPressed) {
                        p->setDirty();
                    }
                }
            }
        }
    });
    if (hotkeyPressed) return true;

    // ------------------------------------------------------------------------

    auto* crv = this->render_view_slot.template CallAs<core::view::CallRenderView>();
    if (crv == nullptr) return false;

    core::view::InputEvent evt;
    evt.tag = core::view::InputEvent::Tag::Key;
    evt.keyData.key = key;
    evt.keyData.action = action;
    evt.keyData.mods = mods;
    crv->SetInputEvent(evt);
    if (!(*crv)(core::view::InputCall::FnOnKey)) return false;

    return false;
}


bool GUIView::OnChar(unsigned int codePoint) {
    ImGui::SetCurrentContext(this->context);

    ImGuiIO& io = ImGui::GetIO();
    io.ClearInputCharacters();
    if (codePoint > 0 && codePoint < 0x10000) io.AddInputCharacter((unsigned short)codePoint);

    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
    if (crv) {
        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::Char;
        evt.charData.codePoint = codePoint;
        crv->SetInputEvent(evt);
        if ((*crv)(core::view::InputCall::FnOnChar)) return true;
    }

    return true;
}


bool GUIView::OnMouseMove(double x, double y) {
    ImGui::SetCurrentContext(this->context);

    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2((float)x, (float)y);

    auto hoverFlags = ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenDisabled |
                      ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem;

    if (!ImGui::IsWindowHovered(hoverFlags)) {
        auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
        if (crv == nullptr) return false;

        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = x;
        evt.mouseMoveData.y = y;
        crv->SetInputEvent(evt);
        if (!(*crv)(core::view::InputCall::FnOnMouseMove)) return false;
    }

    return true;
}


bool GUIView::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {

    ImGui::SetCurrentContext(this->context);

    bool down = (action == core::view::MouseButtonAction::PRESS);
    auto buttonIndex = static_cast<size_t>(button);
    ImGuiIO& io = ImGui::GetIO();

    auto hoverFlags = ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenDisabled |
                      ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem;

    // Trigger saving state when mouse hoverd any window and on button mouse release event
    if ((!down) && (io.MouseDown[buttonIndex]) && hoverFlags) {
        this->state.win_save_state = true;
        this->state.win_save_delay = 0.0f;
    }

    io.MouseDown[buttonIndex] = down;

    if (!ImGui::IsWindowHovered(hoverFlags)) {
        auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
        if (crv == nullptr) return false;

        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;
        crv->SetInputEvent(evt);
        if (!(*crv)(core::view::InputCall::FnOnMouseButton)) return false;
    }

    return true;
}


bool GUIView::OnMouseScroll(double dx, double dy) {
    ImGui::SetCurrentContext(this->context);

    ImGuiIO& io = ImGui::GetIO();
    io.MouseWheelH += (float)dx;
    io.MouseWheel += (float)dy;

    auto hoverFlags = ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenDisabled |
                      ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem;

    if (!ImGui::IsWindowHovered(hoverFlags)) {
        auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
        if (crv == nullptr) return false;

        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseScroll;
        evt.mouseScrollData.dx = dx;
        evt.mouseScrollData.dy = dy;
        crv->SetInputEvent(evt);
        if (!(*crv)(core::view::InputCall::FnOnMouseScroll)) return false;
    }

    return true;
}


bool GUIView::OnRenderView(megamol::core::Call& call) {
    megamol::core::view::CallRenderView* crv = dynamic_cast<megamol::core::view::CallRenderView*>(&call);
    if (crv == nullptr) return false;

    this->overrideCall = crv;

    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));
    context.Time = crv->Time();
    context.InstanceTime = crv->InstanceTime();
    // TODO: Affinity
    this->Render(context);

    this->overrideCall = nullptr;

    return true;
}


void GUIView::validateGUI() {
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

    ImGuiIO& io = ImGui::GetIO();
    this->state.win_save_delay += io.DeltaTime;
    if (this->state_param.IsDirty()) {
        auto state = this->state_param.Param<core::param::StringParam>()->Value();
        this->window_manager.StateFromJSON(std::string(state));
        this->state_param.ResetDirty();
    } else if (this->state.win_save_state &&
               (this->state.win_save_delay > 2.0f)) { // Delayed saving after triggering saving state
        std::string state;
        this->window_manager.StateToJSON(state);
        this->state_param.Param<core::param::StringParam>()->SetValue(state.c_str(), false);
        this->state.win_save_state = false;
    }
}


bool GUIView::drawGUI(vislib::math::Rectangle<int> viewport, double instanceTime) {
    ImGui::SetCurrentContext(this->context);

    this->validateGUI();
    /// So far: Checked only once
    this->checkMultipleHotkeyAssignement();

    auto viewportWidth = viewport.Width();
    auto viewportHeight = viewport.Height();

    // Set IO stuff
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)viewportWidth, (float)viewportHeight);
    io.DisplayFramebufferScale = ImVec2(1.0, 1.0);

    if ((instanceTime - this->state.last_instance_time) < 0.0) {
        vislib::sys::Log::DefaultLog.WriteWarn("[GUIView] Current instance time results in negative time delta.");
    }
    io.DeltaTime = ((instanceTime - this->state.last_instance_time) > 0.0)
                       ? (static_cast<float>(instanceTime - this->state.last_instance_time))
                       : (io.DeltaTime);
    this->state.last_instance_time = ((instanceTime - this->state.last_instance_time) > 0.0)
                                         ? (instanceTime)
                                         : (this->state.last_instance_time + io.DeltaTime);

    // Changes that need to be applied before next ImGui::Begin: ---------------
    // Loading new font (set in FONT window)
    if (!this->state.font_file.empty()) {
        ImFontConfig config;
        config.OversampleH = 4;
        config.OversampleV = 1;
        config.GlyphRanges = this->state.font_utf8_ranges.data();

        this->utils.utf8Encode(this->state.font_file);
        io.Fonts->AddFontFromFileTTF(this->state.font_file.c_str(), this->state.font_size, &config);
        ImGui_ImplOpenGL3_CreateFontsTexture();
        /// Load last added font
        io.FontDefault = io.Fonts->Fonts[(io.Fonts->Fonts.Size - 1)];
        this->state.font_file.clear();
    }

    // Loading new font from state (set in loaded FONT window configuration)
    if (this->state.font_index >= 0) {
        if (this->state.font_index < io.Fonts->Fonts.Size) {
            io.FontDefault = io.Fonts->Fonts[this->state.font_index];
        }
        this->state.font_index = -1;
    }

    // Deleting window (set in menu of MAIN window)
    if (!this->state.win_delete.empty()) {
        this->window_manager.DeleteWindowConfiguration(this->state.win_delete);
        this->state.win_delete.clear();
    }

    // Start new frame --------------------------------------------------------
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();

    const auto func = [&, this](const std::string& wn, WindowManager::WindowConfiguration& wc) {
        // Loading font (from FONT window configuration - even if FONT window is not shown)
        if (wc.buf_font_reset) {
            if (!wc.font_name.empty()) {
                this->state.font_index = -1;
                for (int n = 0; n < io.Fonts->Fonts.Size; n++) {

                    std::string font_name = std::string(io.Fonts->Fonts[n]->GetDebugName());
                    this->utils.utf8Decode(font_name);
                    if (font_name == wc.font_name) {
                        this->state.font_index = n;
                    }
                }
                if (this->state.font_index < 0) {
                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "[GUIView] Could not find font '%s' for loaded state.", wc.font_name.c_str());
                }
            }
            wc.buf_font_reset = false;
        }

        // Draw window content
        if (wc.win_show) {

            ImGui::SetNextWindowBgAlpha(1.0f);
            if (!ImGui::Begin(wn.c_str(), &wc.win_show, wc.win_flags)) {
                ImGui::End(); // early ending
                return;
            }

            // Apply soft reset of window position and size (before calling window callback)
            if (wc.win_soft_reset) {
                this->window_manager.SoftResetWindowSizePos(wn, wc);
                wc.win_soft_reset = false;
            }
            // Apply reset after new state has been loaded (before calling window callback)
            if (wc.buf_win_reset) {
                this->window_manager.ResetWindowOnStateLoad(wn, wc);
                wc.buf_win_reset = false;
            }

            // Calling callback drawing window content
            auto cb = this->window_manager.WindowCallback(wc.win_callback);
            if (cb) {
                cb(wn, wc);
            } else {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[GUIView] Missing valid callback for WindowDrawCallback: '%d'", (int)wc.win_callback);
            }

            // Saving current window position and size for all window configurations for possible state saving.
            wc.win_position = ImGui::GetWindowPos();
            wc.win_size = ImGui::GetWindowSize();

            ImGui::End();
        }
    };
    this->window_manager.EnumWindows(func);

    // Render the frame -------------------------------------------------------
    glViewport(0, 0, viewportWidth, viewportHeight);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return true;
}


void GUIView::drawMainWindowCallback(const std::string& wn, WindowManager::WindowConfiguration& wc) {
    // Menu -------------------------------------------------------------------
    /// Requires window flag ImGuiWindowFlags_MenuBar
    if (ImGui::BeginMenuBar()) {
        this->drawMenu(wn, wc);
        ImGui::EndMenuBar();
    }

    // Parameters -------------------------------------------------------------
    ImGui::Text("Parameters");
    std::string color_param_help = "[Hover] Show Parameter Description Tooltip\n"
                                   "[Right-Click] Context Menu\n"
                                   "[Drag & Drop] Move Module to other Parameter Window\n"
                                   "[Enter],[Tab],[Left-Click outside Widget] Confirm input changes";
    this->utils.HelpMarkerToolTip(color_param_help);

    this->drawParametersCallback(wn, wc);
}


void GUIView::drawTFWindowCallback(const std::string& wn, WindowManager::WindowConfiguration& wc) {
    this->tf_editor.DrawTransferFunctionEditor();
}


void GUIView::drawParametersCallback(const std::string& wn, WindowManager::WindowConfiguration& wc) {
    ImGuiStyle& style = ImGui::GetStyle();
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.5f); // set general proportional item width

    // Options
    int overrideState = -1; /// invalid
    if (ImGui::Button("Expand All")) {
        overrideState = 1; /// open
    }
    ImGui::SameLine();
    if (ImGui::Button("Collapse All")) {
        overrideState = 0; /// close
    }

    bool show_only_hotkeys = wc.param_show_hotkeys;
    ImGui::Checkbox("Show Hotkeys", &show_only_hotkeys);
    wc.param_show_hotkeys = show_only_hotkeys;

    // Offering module filtering only for main parameter view
    if (wc.win_callback == WindowManager::DrawCallbacks::MAIN) {
        std::map<int, std::string> opts;
        opts[static_cast<int>(WindowManager::FilterModes::ALL)] = "All";
        opts[static_cast<int>(WindowManager::FilterModes::INSTANCE)] = "Instance";
        opts[static_cast<int>(WindowManager::FilterModes::VIEW)] = "View";
        unsigned int opts_cnt = (unsigned int)opts.size();
        if (ImGui::BeginCombo("Module Filter", opts[(int)wc.param_module_filter].c_str())) {
            for (unsigned int i = 0; i < opts_cnt; ++i) {

                if (ImGui::Selectable(opts[i].c_str(), (static_cast<int>(wc.param_module_filter) == i))) {
                    wc.param_module_filter = static_cast<WindowManager::FilterModes>(i);
                    wc.param_modules_list.clear();
                    if ((wc.param_module_filter == WindowManager::FilterModes::INSTANCE) ||
                        (wc.param_module_filter == WindowManager::FilterModes::VIEW)) {

                        // Goal is to find view module with shortest call connection path to this module.
                        // Since enumeration of modules goes bottom up, result for first abstract view is
                        // stored and following hits are ignored.
                        std::string viewname;
                        std::string thisname = this->FullName().PeekBuffer();
                        const auto view_func = [&, this](core::Module* viewmod) {
                            auto v = dynamic_cast<core::view::AbstractView*>(viewmod);
                            if (v != nullptr) {
                                std::string vname = v->FullName().PeekBuffer();

                                bool found = false;
                                const auto find_func = [&, this](core::Module* guimod) {
                                    std::string modname = guimod->FullName().PeekBuffer();
                                    if (thisname == modname) {
                                        found = true;
                                    }
                                };
                                this->GetCoreInstance()->EnumModulesNoLock(viewmod, find_func);

                                if (found && viewname.empty()) {
                                    viewname = vname;
                                }
                            }
                        };
                        this->GetCoreInstance()->EnumModulesNoLock(nullptr, view_func);

                        if (!viewname.empty()) {
                            if (wc.param_module_filter == WindowManager::FilterModes::INSTANCE) {
                                // Considering modules depending on the INSTANCE NAME of the first view this module is
                                // connected to.
                                std::string instname = "";
                                if (viewname.find("::", 2) != std::string::npos) {
                                    instname = viewname.substr(0, viewname.find("::", 2));
                                }
                                if (!instname.empty()) { /// Consider all modules if view is not assigned to any
                                                         /// instance
                                    const auto func = [&, this](core::Module* mod) {
                                        std::string modname = mod->FullName().PeekBuffer();
                                        bool foundInstanceName = (modname.find(instname) != std::string::npos);
                                        // Modules with no namespace are always taken into account ...
                                        bool noInstanceNamePresent = (modname.find("::", 2) == std::string::npos);
                                        if (foundInstanceName || noInstanceNamePresent) {
                                            wc.param_modules_list.emplace_back(modname);
                                        }
                                    };
                                    this->GetCoreInstance()->EnumModulesNoLock(nullptr, func);
                                }
                            } else { // (wc.param_module_filter == WindowManager::FilterModes::VIEW)
                                // Considering modules depending on their connection to the first VIEW this module is
                                // connected to.
                                const auto add_func = [&, this](core::Module* mod) {
                                    std::string modname = mod->FullName().PeekBuffer();
                                    wc.param_modules_list.emplace_back(modname);
                                };
                                this->GetCoreInstance()->EnumModulesNoLock(viewname, add_func);
                            }
                        } else {
                            vislib::sys::Log::DefaultLog.WriteWarn("[GUIView] Could not find abstract view "
                                                                   "module this gui is connected to.");
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
        this->utils.HelpMarkerToolTip("Filter applies globally to all parameter windows.\n"
                                      "Selected filter is not refreshed on graph changes.\n"
                                      "Select filter again to trigger refresh.");
        ImGui::Separator();
    }

    // Listing parameters
    const core::Module* current_mod = nullptr;
    bool current_mod_open = false;
    const size_t dnd_size = 2048; // Set same max size of all module labels for drag and drop.
    std::string param_namespace = "";
    unsigned int param_indent_stack = 0;
    bool param_namespace_open = true;

    this->GetCoreInstance()->EnumParameters([&, this](const auto& mod, auto& slot) {
        // Check for new module
        if (current_mod != &mod) {
            current_mod = &mod;
            std::string label = mod.FullName().PeekBuffer();

            // Reset parameter namespace stuff
            param_namespace = "";
            param_namespace_open = true;
            while (param_indent_stack > 0) {
                param_indent_stack--;
                ImGui::Unindent();
            }

            // Check if module should be considered.
            if (!this->considerModule(label, wc.param_modules_list)) {
                current_mod_open = false;
                return;
            }

            // Main parameter window always draws all module's parameters
            if (wc.win_callback != WindowManager::DrawCallbacks::MAIN) {
                // Consider only modules contained in list
                if (std::find(wc.param_modules_list.begin(), wc.param_modules_list.end(), label) ==
                    wc.param_modules_list.end()) {
                    current_mod_open = false;
                    return;
                }
            }

            auto headerId = ImGui::GetID(label.c_str());
            auto headerState = overrideState;
            if (headerState == -1) {
                headerState = ImGui::GetStateStorage()->GetInt(headerId, 0); // 0=close 1=open
            }

            ImGui::GetStateStorage()->SetInt(headerId, headerState);
            current_mod_open = ImGui::CollapsingHeader(label.c_str(), nullptr);

            // TODO:  Add module description as hover tooltip
            // this->utils.HoverToolTip(std::string(mod.Description()), ImGui::GetID(label.c_str()), 0.5f);
            // this->utils.HoverToolTip(std::string(mod.FullName().PeekBuffer()), ImGui::GetID(label.c_str()), 0.5f);

            // Context menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Copy to new Window")) {
                    // using instance time as hidden unique id
                    std::string window_name =
                        "Parameters###parameters" + std::to_string(this->state.last_instance_time);

                    WindowManager::WindowConfiguration buf_win;
                    buf_win.win_show = true;
                    buf_win.win_flags = ImGuiWindowFlags_HorizontalScrollbar;
                    buf_win.win_callback = WindowManager::DrawCallbacks::PARAM;
                    buf_win.param_show_hotkeys = false;
                    buf_win.param_modules_list.emplace_back(label);
                    this->window_manager.AddWindowConfiguration(window_name, buf_win);
                }
                // Deleting module's parameters is not available in main parameter window.
                if (wc.win_callback != WindowManager::DrawCallbacks::MAIN) {
                    if (ImGui::MenuItem("Delete from List")) {
                        std::vector<std::string>::iterator find_iter =
                            std::find(wc.param_modules_list.begin(), wc.param_modules_list.end(), label);
                        // Break if module name is not contained in list
                        if (find_iter != wc.param_modules_list.end()) {
                            wc.param_modules_list.erase(find_iter);
                        }
                    }
                }
                ImGui::EndPopup();
            }

            // Drag source
            label.resize(dnd_size);
            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
                ImGui::SetDragDropPayload("DND_COPY_MODULE_PARAMETERS", label.c_str(), (label.size() * sizeof(char)));
                ImGui::Text(label.c_str());
                ImGui::EndDragDropSource();
            }
        }

        if (current_mod_open) {
            auto param = slot.Parameter();
            if (!param.IsNull() && param->IsGUIVisible()) {

                // Check for new parameter namespace
                std::string param_name = slot.Name().PeekBuffer();
                auto pos = param_name.find("::");
                std::string current_param_namespace = "";
                if (pos != std::string::npos) {
                    current_param_namespace = param_name.substr(0, pos);
                }
                if (current_param_namespace != param_namespace) {

                    param_namespace = current_param_namespace;

                    while (param_indent_stack > 0) {
                        param_indent_stack--;
                        ImGui::Unindent();
                    }

                    ImGui::Separator();
                    if (!param_namespace.empty()) {
                        ImGui::Indent();
                        std::string label = param_namespace + "###" + param_namespace + "__" + param_name;
                        param_namespace_open = ImGui::CollapsingHeader(label.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
                        param_indent_stack++;
                    } else {
                        param_namespace_open = true;
                    }
                }

                // Draw parameter
                if (param_namespace_open) {
                    if (wc.param_show_hotkeys) {
                        this->drawParameterHotkey(mod, slot);
                    } else {
                        this->drawParameter(mod, slot);
                    }
                }
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

            // Nothing to add to main parameter window (draws always all module's parameters)
            if ((wc.win_callback != WindowManager::DrawCallbacks::MAIN)) {
                // Insert dragged module name only if not contained in list
                if (std::find(wc.param_modules_list.begin(), wc.param_modules_list.end(), payload_id) ==
                    wc.param_modules_list.end()) {
                    wc.param_modules_list.emplace_back(payload_id);
                }
            }
        }
        ImGui::EndDragDropTarget();
    }

    ImGui::PopItemWidth();
}


void GUIView::drawFpsWindowCallback(const std::string& wn, WindowManager::WindowConfiguration& wc) {
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    wc.buf_current_delay += io.DeltaTime;
    if (wc.fpsms_refresh_rate <= 0.0f) {
        return;
    }
    if (wc.buf_max_history_count == 0) {
        wc.buf_fps_values.clear();
        wc.buf_ms_values.clear();
        return;
    }

    if (wc.buf_current_delay > (1.0f / wc.fpsms_refresh_rate)) {
        // Leave some space in histogram for text of current value
        const float scale_fac = 1.5f;

        if (wc.buf_fps_values.size() != wc.buf_ms_values.size()) {
            vislib::sys::Log::DefaultLog.WriteError(
                "[GUIView] Arrays for FPS and frame times do not have equal length.");
            return;
        }

        int size = (int)wc.buf_fps_values.size();
        if (size != wc.fpsms_max_history_count) {
            if (size > wc.fpsms_max_history_count) {
                wc.buf_fps_values.erase(
                    wc.buf_fps_values.begin(), wc.buf_fps_values.begin() + (size - wc.fpsms_max_history_count));
                wc.buf_ms_values.erase(
                    wc.buf_ms_values.begin(), wc.buf_ms_values.begin() + (size - wc.fpsms_max_history_count));

            } else if (size < wc.fpsms_max_history_count) {
                wc.buf_fps_values.insert(wc.buf_fps_values.begin(), (wc.fpsms_max_history_count - size), 0.0f);
                wc.buf_ms_values.insert(wc.buf_ms_values.begin(), (wc.fpsms_max_history_count - size), 0.0f);
            }
        }
        if (size > 0) {
            wc.buf_fps_values.erase(wc.buf_fps_values.begin());
            wc.buf_ms_values.erase(wc.buf_ms_values.begin());

            wc.buf_fps_values.emplace_back(io.Framerate);
            wc.buf_ms_values.emplace_back(io.DeltaTime * 1000.0f); // scale to milliseconds

            float value_max = 0.0f;
            for (auto& v : wc.buf_fps_values) {
                value_max = (v > value_max) ? (v) : (value_max);
            }
            wc.buf_fps_scale = value_max * scale_fac;

            value_max = 0.0f;
            for (auto& v : wc.buf_ms_values) {
                value_max = (v > value_max) ? (v) : (value_max);
            }
            wc.buf_ms_scale = value_max * scale_fac;
        }

        wc.buf_current_delay = 0.0f;
    }

    // Draw window content
    if (ImGui::RadioButton("fps", (wc.fpsms_mode == WindowManager::TimingModes::FPS))) {
        wc.fpsms_mode = WindowManager::TimingModes::FPS;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("ms", (wc.fpsms_mode == WindowManager::TimingModes::MS))) {
        wc.fpsms_mode = WindowManager::TimingModes::MS;
    }

    ImGui::SameLine(0.0f, 50.0f);
    ImGui::Checkbox("Options", &wc.fpsms_show_options);

    // Default for wc.fpsms_mode == WindowManager::TimingModes::FPS
    std::vector<float>* arr = &wc.buf_fps_values;
    float val_scale = wc.buf_fps_scale;
    if (wc.fpsms_mode == WindowManager::TimingModes::MS) {
        arr = &wc.buf_ms_values;
        val_scale = wc.buf_ms_scale;
    }
    float* data = arr->data();
    int count = (int)arr->size();

    std::string val;
    if (!arr->empty()) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(3) << arr->back();
        val = stream.str();
    }
    ImGui::PlotLines(
        "###fpsmsplot", data, count, 0, val.c_str(), 0.0f, val_scale, ImVec2(0.0f, 50.0f)); /// use hidden label
    float item_width = ImGui::GetItemRectSize().x;

    if (wc.fpsms_show_options) {

        // Refresh rate
        ImGui::InputFloat("Refresh Rate", &wc.buf_refresh_rate, 1.0f, 10.0f, "%.3f", ImGuiInputTextFlags_None);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            wc.fpsms_refresh_rate = std::max(0.0f, wc.buf_refresh_rate);
            wc.buf_fps_values.clear();
            wc.buf_ms_values.clear();
            wc.buf_refresh_rate = wc.fpsms_refresh_rate;
        }
        std::string help = "Changes clear all values";
        this->utils.HelpMarkerToolTip(help);

        // History
        ImGui::InputInt("History Size", &wc.buf_max_history_count, 1, 10, ImGuiInputTextFlags_None);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            wc.fpsms_max_history_count = std::max(1, wc.buf_max_history_count);
            wc.buf_max_history_count = wc.fpsms_max_history_count;
        }

        if (ImGui::Button("Current Value")) {
            ImGui::SetClipboardText(val.c_str());
        }
        ImGui::SameLine();

        if (ImGui::Button("All Values")) {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(3);
            auto end = (*arr).rend();
            for (std::vector<float>::reverse_iterator i = (*arr).rbegin(); i != end; ++i) {
                stream << (*i) << "\n";
            }
            ImGui::SetClipboardText(stream.str().c_str());
        }
        ImGui::SameLine();
        ImGui::SetCursorPosX(item_width + style.ItemSpacing.x + style.ItemInnerSpacing.x);
        ImGui::Text("Copy to Clipborad");
        help = "Values are copied in chronological order (newest first)";
        this->utils.HelpMarkerToolTip(help);
    }
}


void GUIView::drawFontWindowCallback(const std::string& wn, WindowManager::WindowConfiguration& wc) {
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    ImFont* font_current = ImGui::GetFont();
    if (ImGui::BeginCombo("Select available Font", font_current->GetDebugName())) {
        for (int n = 0; n < io.Fonts->Fonts.Size; n++) {
            if (ImGui::Selectable(io.Fonts->Fonts[n]->GetDebugName(), (io.Fonts->Fonts[n] == font_current)))
                io.FontDefault = io.Fonts->Fonts[n];
        }
        ImGui::EndCombo();
    }

    // Saving current font to window configuration.
    wc.font_name = std::string(font_current->GetDebugName());
    this->utils.utf8Decode(wc.font_name);

#ifdef GUI_USE_FILEUTILS
    ImGui::Separator();
    ImGui::Text("Load Font from File");
    std::string help = "Same font can be loaded multiple times with different font size.";
    this->utils.HelpMarkerToolTip(help);

    std::string label = "Font Size";
    ImGui::InputFloat(label.c_str(), &wc.buf_font_size, 1.0f, 10.0f, "%.2f", ImGuiInputTextFlags_None);
    // Validate font size
    if (wc.buf_font_size <= 0.0f) {
        wc.buf_font_size = 5.0f; /// min valid font size
    }

    label = "Font File Name (.ttf)";
    /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
    this->utils.utf8Encode(wc.buf_font_file);
    ImGui::InputText(label.c_str(), &wc.buf_font_file, ImGuiInputTextFlags_AutoSelectAll);
    this->utils.utf8Decode(wc.buf_font_file);
    // Validate font file before offering load button
    if (HasExistingFileExtension(wc.buf_font_file, std::string(".ttf"))) {
        if (ImGui::Button("Add Font")) {
            this->state.font_file = wc.buf_font_file;
            this->state.font_size = wc.buf_font_size;
        }
    } else {
        ImGui::TextColored(ImVec4(0.9f, 0.0f, 0.0f, 1.0f), "Please enter valid font file name.");
    }
#endif // GUI_USE_FILEUTILS
}


void GUIView::drawMenu(const std::string& wn, WindowManager::WindowConfiguration& wc) {
    ImGuiStyle& style = ImGui::GetStyle();

    bool open_popup_project = false;
    if (ImGui::BeginMenu("File")) {
#ifdef GUI_USE_FILEUTILS
        // Load/save parameter values to LUA file
        if (ImGui::MenuItem("Save Project")) {
            open_popup_project = true;
        }
        /// Not supported so far
        // if (ImGui::MenuItem("Load Project")) {
        //    // TODO:  Load parameter file
        //    std::string projectFilename;
        //    this->GetCoreInstance()->LoadProject(vislib::StringA(projectFilename.c_str()));
        //}
#endif // GUI_USE_FILEUTILS

        if (ImGui::MenuItem("Exit", "ALT + 'F4'")) {
            // Exit program
            this->shutdown();
        }
        ImGui::EndMenu();
    }

    // Windows
    if (ImGui::BeginMenu("Windows")) {
        const auto func = [&, this](const std::string& wn, WindowManager::WindowConfiguration& wc) {
            bool win_open = wc.win_show;
            std::string hotkey_label = wc.win_hotkey.ToString();
            if (!hotkey_label.empty()) {
                hotkey_label = "(SHIFT +) " + hotkey_label;
            }
            if (ImGui::MenuItem(wn.c_str(), hotkey_label.c_str(), &win_open)) {
                wc.win_show = !wc.win_show;
            }
            // Add conext menu for deleting windows without hotkey (= custom parameter windows).
            if (wc.win_hotkey.GetKey() == core::view::Key::KEY_UNKNOWN) {
                if (ImGui::BeginPopupContextItem()) {
                    if (ImGui::MenuItem("Delete Window")) {
                        this->state.win_delete = wn;
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
    bool open_popup_about = false;
    if (ImGui::BeginMenu("Help")) {
        // if (ImGui::MenuItem("Usability Hints")) {
        //}

        if (ImGui::MenuItem("About")) {
            open_popup_about = true;
        }
        ImGui::EndMenu();
    }

    // Popups -----------------------------------------------------------------

    // ABOUT
    if (open_popup_about) {
        ImGui::OpenPopup("About");
    }
    bool open = true;
    if (ImGui::BeginPopupModal("About", &open, ImGuiWindowFlags_AlwaysAutoResize)) {

        const std::string eMail = "megamol@visus.uni-stuttgart.de";
        const std::string webLink = "https://megamol.org/";
        const std::string gitLink = "https://github.com/UniStuttgart-VISUS/megamol";

        std::string about = std::string("MegaMol - Version ") + std::to_string(MEGAMOL_CORE_MAJOR_VER) + (".") +
                            std::to_string(MEGAMOL_CORE_MINOR_VER) + ("\ngit# ") + std::string(MEGAMOL_CORE_COMP_REV) +
                            ("\nDear ImGui - Version ") + std::string(IMGUI_VERSION) + ("\n");
        std::string mailstr = std::string("Contact: ") + eMail;
        std::string webstr = std::string("Web: ") + webLink;
        std::string gitstr = std::string("Git-Hub: ") + gitLink;

        ImGui::Text(about.c_str());
        ImGui::Separator();

        if (ImGui::Button("Copy E-Mail")) {
            ImGui::SetClipboardText(eMail.c_str());
        }
        ImGui::SameLine();
        ImGui::Text(mailstr.c_str());

        if (ImGui::Button("Copy Website")) {
            ImGui::SetClipboardText(webLink.c_str());
        }
        ImGui::SameLine();
        ImGui::Text(webstr.c_str());

        if (ImGui::Button("Copy GitHub")) {
            ImGui::SetClipboardText(gitLink.c_str());
        }
        ImGui::SameLine();
        ImGui::Text(gitstr.c_str());

        ImGui::Separator();
        about = "Copyright (C) 2009-2019 by Universitaet Stuttgart "
                "(VIS).\nAll rights reserved.";
        ImGui::Text(about.c_str());

        ImGui::Separator();
        if (ImGui::Button("Close")) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
    }

    // SAVE PROJECT
#ifdef GUI_USE_FILEUTILS
    if (open_popup_project) {
        ImGui::OpenPopup("Save Project");
    }
    if (ImGui::BeginPopupModal("Save Project", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {

        std::string label = "File Name";
        /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
        this->utils.utf8Encode(wc.main_project_file);
        ImGui::InputText(label.c_str(), &wc.main_project_file, ImGuiInputTextFlags_None);
        this->utils.utf8Decode(wc.main_project_file);

        bool valid = true;
        if (!HasFileExtension(wc.main_project_file, std::string(".lua"))) {
            ImGui::TextColored(ImVec4(0.9f, 0.0f, 0.0f, 1.0f), "File name needs to have the ending '.lua'");
            valid = false;
        }
        // Warn when file already exists
        if (PathExists(wc.main_project_file)) {
            ImGui::TextColored(ImVec4(0.9f, 0.0f, 0.0f, 1.0f), "File name already exists and will be overwritten.");
        }
        if (ImGui::Button("Save")) {
            if (valid) {
                // Serialize current state to parameter.
                std::string state;
                this->window_manager.StateToJSON(state);
                this->state_param.Param<core::param::StringParam>()->SetValue(state.c_str(), false);
                // Save project to file
                if (SaveProjectFile(wc.main_project_file, this->GetCoreInstance())) {
                    ImGui::CloseCurrentPopup();
                }
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
#endif // GUI_USE_FILEUTILS
}


void GUIView::drawParameter(const core::Module& mod, core::param::ParamSlot& slot) {
    ImGuiStyle& style = ImGui::GetStyle();
    std::string help;

    auto param = slot.Parameter();
    if (!param.IsNull()) {
        // Set different style if parameter is read-only
        if (param->IsGUIReadOnly()) {
            ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
        }

        std::string param_name = slot.Name().PeekBuffer();
        std::string param_id = std::string(mod.FullName().PeekBuffer()) + "::" + param_name;
        auto pos = param_name.find("::");
        if (pos != std::string::npos) {
            param_name = param_name.substr(pos + 2);
        }
        std::string param_label_hidden = "###" + param_id;
        std::string param_label = param_name + param_label_hidden;
        std::string param_desc = slot.Description().PeekBuffer();
        std::string float_format = "%.7f";

        if (auto* p = slot.template Param<core::param::BoolParam>()) {
            auto value = p->Value();
            if (ImGui::Checkbox(param_label.c_str(), &value)) {
                p->SetValue(value);
            }
        } else if (auto* p = slot.template Param<core::param::ButtonParam>()) {
            std::string hotkey = "";
            std::string buttonHotkey = p->GetKeyCode().ToString();
            if (!buttonHotkey.empty()) {
                hotkey = " (" + buttonHotkey + ")";
            }
            auto insert_pos = param_label.find("###");
            if (insert_pos == std::string::npos) {
                param_label.insert(insert_pos, hotkey);
            }
            if (ImGui::Button(param_label.c_str())) {
                p->setDirty();
            }
        } else if (auto* p = slot.template Param<core::param::ColorParam>()) {
            core::param::ColorParam::ColorType value = p->Value();
            auto color_flags = ImGuiColorEditFlags_AlphaPreview; // | ImGuiColorEditFlags_Float;
            if (ImGui::ColorEdit4(param_label.c_str(), (float*)value.data(), color_flags)) {
                p->SetValue(value);
            }
            help = "[Click] on the colored square to open a color picker.\n"
                   "[CTRL+Click] on individual component to input value.\n"
                   "[Right-Click] on the individual color widget to show options.";
        } else if (auto* p = slot.template Param<core::param::TransferFunctionParam>()) {
            drawTransferFunctionEdit(param_id, param_label, *p);
        } else if (auto* p = slot.template Param<core::param::EnumParam>()) {
            /// XXX: no UTF8 fanciness required here?
            auto map = p->getMap();
            auto key = p->Value();
            if (ImGui::BeginCombo(param_label.c_str(), map[key].PeekBuffer())) {
                auto iter = map.GetConstIterator();
                while (iter.HasNext()) {
                    auto pair = iter.Next();
                    bool isSelected = (pair.Key() == key);
                    if (ImGui::Selectable(pair.Value().PeekBuffer(), isSelected)) {
                        p->SetValue(pair.Key());
                    }
                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
        } else if (auto* p = slot.template Param<core::param::FlexEnumParam>()) {
            /// XXX: no UTF8 fanciness required here?
            auto value = p->Value();
            if (ImGui::BeginCombo(param_label.c_str(), value.c_str())) {
                for (auto valueOption : p->getStorage()) {
                    bool isSelected = (valueOption == value);
                    if (ImGui::Selectable(valueOption.c_str(), isSelected)) {
                        p->SetValue(valueOption);
                    }
                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
        } else if (auto* p = slot.template Param<core::param::FloatParam>()) {
            auto it = this->widgtmap_float.find(param_id);
            if (it == this->widgtmap_float.end()) {
                this->widgtmap_float.emplace(param_id, p->Value());
                it = this->widgtmap_float.find(param_id);
            }
            ImGui::InputFloat(
                param_label.c_str(), &it->second, 1.0f, 10.0f, float_format.c_str(), ImGuiInputTextFlags_None);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                p->SetValue(std::max(p->MinValue(), std::min(it->second, p->MaxValue())));
            } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                it->second = p->Value();
            }
        } else if (auto* p = slot.template Param<core::param::IntParam>()) {
            auto it = this->widgtmap_int.find(param_id);
            if (it == this->widgtmap_int.end()) {
                this->widgtmap_int.emplace(param_id, p->Value());
                it = this->widgtmap_int.find(param_id);
            }
            ImGui::InputInt(param_label.c_str(), &it->second, 1, 10, ImGuiInputTextFlags_None);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                p->SetValue(std::max(p->MinValue(), std::min(it->second, p->MaxValue())));
            } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                it->second = p->Value();
            }
        } else if (auto* p = slot.template Param<core::param::Vector2fParam>()) {
            auto it = this->widgtmap_vec2.find(param_id);
            if (it == this->widgtmap_vec2.end()) {
                this->widgtmap_vec2.emplace(param_id, p->Value());
                it = this->widgtmap_vec2.find(param_id);
            }
            ImGui::InputFloat2(
                param_label.c_str(), it->second.PeekComponents(), float_format.c_str(), ImGuiInputTextFlags_None);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                p->SetValue(it->second);
            } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                it->second = p->Value();
            }
        } else if (auto* p = slot.template Param<core::param::Vector3fParam>()) {
            auto it = this->widgtmap_vec3.find(param_id);
            if (it == this->widgtmap_vec3.end()) {
                this->widgtmap_vec3.emplace(param_id, p->Value());
                it = this->widgtmap_vec3.find(param_id);
            }
            ImGui::InputFloat3(
                param_label.c_str(), it->second.PeekComponents(), float_format.c_str(), ImGuiInputTextFlags_None);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                p->SetValue(it->second);
            } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                it->second = p->Value();
            }
        } else if (auto* p = slot.template Param<core::param::Vector4fParam>()) {
            auto it = this->widgtmap_vec4.find(param_id);
            if (it == this->widgtmap_vec4.end()) {
                this->widgtmap_vec4.emplace(param_id, p->Value());
                it = this->widgtmap_vec4.find(param_id);
            }
            ImGui::InputFloat4(
                param_label.c_str(), it->second.PeekComponents(), float_format.c_str(), ImGuiInputTextFlags_None);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                p->SetValue(it->second);
            } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                it->second = p->Value();
            }
        } else if (auto* p = slot.Param<core::param::StringParam>()) {
            /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
            auto it = this->widgtmap_text.find(param_id);
            if (it == this->widgtmap_text.end()) {
                std::string utf8Str = std::string(p->ValueString().PeekBuffer());
                this->utils.utf8Encode(utf8Str);
                this->widgtmap_text.emplace(param_id, utf8Str);
                it = this->widgtmap_text.find(param_id);
            }
            // Determine multi line count of string
            int lcnt = static_cast<int>(std::count(it->second.begin(), it->second.end(), '\n'));
            lcnt = std::min(static_cast<int>(GUI_MAX_MULITLINE), lcnt);
            ImVec2 ml_dim = ImVec2(
                ImGui::CalcItemWidth(), ImGui::GetFrameHeight() + (ImGui::GetFontSize() * static_cast<float>(lcnt)));
            ImGui::InputTextMultiline(
                param_label_hidden.c_str(), &it->second, ml_dim, ImGuiInputTextFlags_CtrlEnterForNewLine);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                std::string utf8Str = it->second;
                this->utils.utf8Decode(utf8Str);
                p->SetValue(vislib::StringA(utf8Str.c_str()));
            } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                std::string utf8Str = std::string(p->ValueString().PeekBuffer());
                this->utils.utf8Encode(utf8Str);
                it->second = utf8Str;
            }
            ImGui::SameLine();
            ImGui::Text(param_name.c_str());
            help = "[Ctrl + Enter] for new line.\nPress [Return] to confirm changes.";
        } else if (auto* p = slot.Param<core::param::FilePathParam>()) {
            /// XXX: UTF8 conversion and allocation every frame is horrific inefficient.
            auto it = this->widgtmap_text.find(param_id);
            if (it == this->widgtmap_text.end()) {
                std::string utf8Str = std::string(p->ValueString().PeekBuffer());
                this->utils.utf8Encode(utf8Str);
                this->widgtmap_text.emplace(param_id, utf8Str);
                it = this->widgtmap_text.find(param_id);
            }
            ImGui::InputText(param_label.c_str(), &it->second, ImGuiInputTextFlags_None);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                this->utils.utf8Decode(it->second);
                p->SetValue(vislib::StringA(it->second.c_str()));
            } else if (!ImGui::IsItemActive() && !ImGui::IsItemEdited()) {
                std::string utf8Str = std::string(p->ValueString().PeekBuffer());
                this->utils.utf8Encode(utf8Str);
                it->second = utf8Str;
            }
        } else {
            vislib::sys::Log::DefaultLog.WriteWarn("[GUIView] Unknown Parameter Type.");
            return;
        }

        // Reset to default style
        if (param->IsGUIReadOnly()) {
            ImGui::PopItemFlag();
            ImGui::PopStyleVar();
        }

        this->utils.HoverToolTip(param_desc, ImGui::GetID(param_label.c_str()), 0.5f);
        this->utils.HelpMarkerToolTip(help);
    }
}

void GUIView::drawTransferFunctionEdit(
    const std::string& id, const std::string& label, megamol::core::param::TransferFunctionParam& p) {
    ImGuiStyle& style = ImGui::GetStyle();

    ImGui::BeginGroup();
    ImGui::PushID(id.c_str());

    // Reduced display of value and editor state.
    if (p.Value().empty()) {
        ImGui::TextDisabled("{    (empty)    }");
    } else {
        // XXX: A gradient texture would be nice here (sharing some editor code?)
        ImGui::Text("{ ............. }");
    }

    bool isActive = (&p == this->tf_editor.GetActiveParameter());
    bool updateEditor = false;

    // Copy transfer function.
    if (ImGui::Button("Copy")) {
        ImGui::SetClipboardText(p.Value().c_str());
    }

    //  Paste transfer function.
    ImGui::SameLine();
    if (ImGui::Button("Paste")) {
        p.SetValue(ImGui::GetClipboardText());
        updateEditor = true;
    }

    // Edit transfer function.
    ImGui::SameLine();
    ImGui::PushID("Edit_");
    ImGui::PushStyleColor(ImGuiCol_Button, style.Colors[isActive ? ImGuiCol_ButtonHovered : ImGuiCol_Button]);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, style.Colors[isActive ? ImGuiCol_Button : ImGuiCol_ButtonHovered]);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, style.Colors[ImGuiCol_ButtonActive]);
    if (ImGui::Button("Edit")) {
        updateEditor = true;
        isActive = true;
        this->tf_editor.SetActiveParameter(&p);
        // Open window calling the transfer function editor callback
        const auto func = [&, this](const std::string& wn, WindowManager::WindowConfiguration& wc) {
            if (wc.win_callback == WindowManager::DrawCallbacks::TF) {
                wc.win_show = true;
            }
        };
        this->window_manager.EnumWindows(func);
    }
    ImGui::PopStyleColor(3);
    ImGui::PopID();

    // Propagate the transfer function to the editor.
    if (isActive && updateEditor) {
        this->tf_editor.SetTransferFunction(p.Value());
    }

    ImGui::PopID();

    ImGui::SameLine();
    ImGui::TextEx(label.c_str(), ImGui::FindRenderedTextEnd(label.c_str()));
    ImGui::EndGroup();
}


void GUIView::drawParameterHotkey(const core::Module& mod, core::param::ParamSlot& slot) {
    auto param = slot.Parameter();
    if (!param.IsNull()) {
        if (auto* p = slot.template Param<core::param::ButtonParam>()) {
            std::string label = slot.Name().PeekBuffer();
            std::string desc = slot.Description().PeekBuffer();
            std::string keycode = p->GetKeyCode().ToString();

            ImGui::Columns(2, "hotkey_columns", false);

            ImGui::Text(label.c_str());
            this->utils.HoverToolTip(desc);

            ImGui::NextColumn();

            ImGui::Text(keycode.c_str());
            this->utils.HoverToolTip(desc);

            // Reset colums
            ImGui::Columns(1);

            ImGui::Separator();
        }
    }
}


bool GUIView::considerModule(const std::string& modname, std::vector<std::string>& modules_list) {
    bool retval = false;
    if (modules_list.empty()) {
        retval = true;
    } else {
        for (auto mod : modules_list) {
            if (modname == mod) {
                retval = true;
                break;
            }
        }
    }
    return retval;
}


void GUIView::checkMultipleHotkeyAssignement(void) {
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

        this->GetCoreInstance()->EnumParameters([&, this](const auto& mod, auto& slot) {
            auto param = slot.Parameter();
            if (!param.IsNull()) {
                if (auto* p = slot.template Param<core::param::ButtonParam>()) {
                    auto hotkey = p->GetKeyCode();

                    // Ignore not set hotekey
                    if (hotkey.GetKey() == core::view::Key::KEY_UNKNOWN) {
                        return;
                    }

                    // check in hotkey map
                    bool found = false;
                    for (auto kc : hotkeylist) {
                        if ((kc.GetKey() == hotkey.GetKey()) && (kc.GetModifiers().equals(hotkey.GetModifiers()))) {
                            found = true;
                        }
                    }
                    if (!found) {
                        hotkeylist.emplace_back(hotkey);
                    } else {
                        vislib::sys::Log::DefaultLog.WriteWarn(
                            "[GUIView] The hotkey [%s] of the parameter \"%s::%s\" has already been assigned. "
                            ">>> If this hotkey is pressed, there will be no effect on this parameter!",
                            hotkey.ToString().c_str(), mod.FullName().PeekBuffer(), slot.Name().PeekBuffer());
                    }
                }
            }
        });

        this->state.hotkeys_check_once = false;
    }
}


void GUIView::shutdown(void) {
    vislib::sys::Log::DefaultLog.WriteInfo("[GUIView] Triggering MegaMol instance shutdown...");
    this->GetCoreInstance()->Shutdown();
}
