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
 * - Quit program:      Esc, Alt + F4
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

#include "vislib/UTF8Encoder.h"

#include <imgui_internal.h>
#include "CorporateGreyStyle.h"
#include "CorporateWhiteStyle.h"
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"

#include <iomanip>
#include <sstream>

using namespace megamol;
using namespace megamol::gui;
using vislib::sys::Log;

enum Styles {
    CorporateGray,
    CorporateWhite,
    DarkColors,
    LightColors,
};

GUIView::GUIView()
    : core::view::AbstractView()
    , renderViewSlot("renderview", "Connects to a preceding RenderView that will be decorated with a GUI")
    , styleParam("style", "Color style, i.e., theme")
    , stateParam("state", "Current state of all windows")
    , context(nullptr)
    , windowManager()
    , tfEditor()
    , lastInstanceTime(0.0)
    , fontUtf8Ranges()
    , projectFilename()
    , newFontFilenameToLoad()
    , newFontSizeToLoad(13.0f)
    , newFontIndexToLoad(-1)
    , windowToDelete()
    , saveState(false)
    , saveStateDelay(0.0f)
    , checkHotkeysOnce(true) {

    this->renderViewSlot.SetCompatibleCall<core::view::CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->renderViewSlot);

    core::param::EnumParam* styles = new core::param::EnumParam(2);
    styles->SetTypePair(CorporateGray, "Corporate Gray");
    styles->SetTypePair(CorporateWhite, "Corporate White");
    styles->SetTypePair(DarkColors, "Dark Colors");
    styles->SetTypePair(LightColors, "Light Colors");
    this->styleParam << styles;
    this->styleParam.ForceSetDirty();
    this->MakeSlotAvailable(&this->styleParam);

    this->stateParam << new core::param::StringParam("");
    this->MakeSlotAvailable(&this->stateParam);
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
    this->windowManager.RegisterDrawWindowCallback(WindowManager::WindowDrawCallback::MAIN,
        [&, this](const std::string& window_name, WindowManager::WindowConfiguration& window_config) {
            this->drawMainWindowCallback(window_name, window_config);
        });
    this->windowManager.RegisterDrawWindowCallback(WindowManager::WindowDrawCallback::PARAM,
        [&, this](const std::string& window_name, WindowManager::WindowConfiguration& window_config) {
            this->drawParametersCallback(window_name, window_config);
        });
    this->windowManager.RegisterDrawWindowCallback(WindowManager::WindowDrawCallback::FPSMS,
        [&, this](const std::string& window_name, WindowManager::WindowConfiguration& window_config) {
            this->drawFpsWindowCallback(window_name, window_config);
        });
    this->windowManager.RegisterDrawWindowCallback(WindowManager::WindowDrawCallback::FONT,
        [&, this](const std::string& window_name, WindowManager::WindowConfiguration& window_config) {
            this->drawFontWindowCallback(window_name, window_config);
        });
    this->windowManager.RegisterDrawWindowCallback(WindowManager::WindowDrawCallback::TF,
        [&, this](const std::string& window_name, WindowManager::WindowConfiguration& window_config) {
            this->drawTFWindowCallback(window_name, window_config);
        });

    // Create window configurations
    WindowManager::WindowConfiguration tmp_win;
    // MAIN Window ------------------------------------------------------------
    tmp_win.win_show = true;
    tmp_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F12, core::view::Modifier::CTRL);
    tmp_win.win_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_HorizontalScrollbar | ImGuiWindowFlags_NoTitleBar;
    tmp_win.win_callback = WindowManager::WindowDrawCallback::MAIN;
    tmp_win.win_position = ImVec2(12, 12);
    tmp_win.win_size = ImVec2(250, 600);
    tmp_win.win_reset = true;
    this->windowManager.AddWindowConfiguration("MegaMol", tmp_win);

    // FPS/MS Window ----------------------------------------------------------
    tmp_win.win_show = false;
    tmp_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F11, core::view::Modifier::CTRL);
    tmp_win.win_flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar;
    tmp_win.win_callback = WindowManager::WindowDrawCallback::FPSMS;
    this->windowManager.AddWindowConfiguration("Performance Metrics", tmp_win);

    // FONT Window ------------------------------------------------------------
    tmp_win.win_show = false;
    tmp_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F10, core::view::Modifier::CTRL);
    tmp_win.win_flags = ImGuiWindowFlags_AlwaysAutoResize;
    tmp_win.win_callback = WindowManager::WindowDrawCallback::FONT;
    this->windowManager.AddWindowConfiguration("Font Settings", tmp_win);

    // TRANSFER FUNCTION Window -----------------------------------------------
    tmp_win.win_show = false;
    tmp_win.win_hotkey = core::view::KeyCode(core::view::Key::KEY_F9, core::view::Modifier::CTRL);
    tmp_win.win_flags = ImGuiWindowFlags_AlwaysAutoResize;
    tmp_win.win_callback = WindowManager::WindowDrawCallback::TF;
    this->windowManager.AddWindowConfiguration("Transfer Function Editor", tmp_win);

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

    // Adding additional utf-8 glyph ranges
    /// (there is no error if glyph has no representation in font atlas)
    this->fontUtf8Ranges.emplace_back(0x0020);
    this->fontUtf8Ranges.emplace_back(0x00FF); // Basic Latin + Latin Supplement
    this->fontUtf8Ranges.emplace_back(0x20AC);
    this->fontUtf8Ranges.emplace_back(0x20AC); // Euro
    this->fontUtf8Ranges.emplace_back(0x2122);
    this->fontUtf8Ranges.emplace_back(0x2122); // TM
    this->fontUtf8Ranges.emplace_back(0x212B);
    this->fontUtf8Ranges.emplace_back(0x212B); // Angstroem
    this->fontUtf8Ranges.emplace_back(0x0391);
    this->fontUtf8Ranges.emplace_back(0x03D6); // greek alphabet
    this->fontUtf8Ranges.emplace_back(0);      // (range termination)

    // Load initial fonts only once for all imgui contexts
    if (!other_context) {
        ImFontConfig config;
        config.OversampleH = 6;
        config.GlyphRanges = this->fontUtf8Ranges.data();
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
    auto* crv = this->renderViewSlot.CallAs<core::view::CallRenderView>();
    if (crv) {
        crv->SetOutputBuffer(GL_BACK);
        crv->SetInstanceTime(context.InstanceTime);
        crv->SetTime(
            -1.0f); // Should be negative to trigger animation! (see View3D.cpp line ~660 | View2D.cpp line ~350)
        (*crv)(core::view::AbstractCallRender::FnRender);
    } else {
        ::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        ::glClear(GL_COLOR_BUFFER_BIT);
    }
    this->drawGUI(crv->GetViewport(), crv->InstanceTime());
}


void GUIView::ResetView(void) {
    auto* crv = this->renderViewSlot.CallAs<core::view::CallRenderView>();
    if (crv) {
        (*crv)(core::view::CallRenderView::CALL_RESETVIEW);
    }
}


void GUIView::Resize(unsigned int width, unsigned int height) {
    auto* crv = this->renderViewSlot.CallAs<core::view::CallRenderView>();
    if (crv) {
        // der ganz ganz dicke "because-i-know"-Knueppel
        AbstractView* view = const_cast<AbstractView*>(
            dynamic_cast<const AbstractView*>(static_cast<const Module*>(crv->PeekCalleeSlot()->Owner())));
        if (view != NULL) {
            view->Resize(width, height);
        }
    }
}


void GUIView::UpdateFreeze(bool freeze) {
    auto* crv = this->renderViewSlot.CallAs<core::view::CallRenderView>();
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
    hotkeyPressed = (ImGui::IsKeyDown(io.KeyMap[ImGuiKey_Escape])) ||                               // Escape
                    ((io.KeyAlt) && (ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_F4)))); // Alt + F4
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
    this->windowManager.EnumWindows(func);

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
    this->windowManager.EnumWindows(modfunc);
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

    auto* crv = this->renderViewSlot.template CallAs<core::view::CallRenderView>();
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

    auto* crv = this->renderViewSlot.CallAs<core::view::CallRenderView>();
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
        auto* crv = this->renderViewSlot.CallAs<core::view::CallRenderView>();
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
        this->saveState = true;
        this->saveStateDelay = 0.0f;
    }

    io.MouseDown[buttonIndex] = down;

    if (!ImGui::IsWindowHovered(hoverFlags)) {
        auto* crv = this->renderViewSlot.CallAs<core::view::CallRenderView>();
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
        auto* crv = this->renderViewSlot.CallAs<core::view::CallRenderView>();
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


void GUIView::validateGUI() {
    if (this->styleParam.IsDirty()) {
        auto style = this->styleParam.Param<core::param::EnumParam>()->Value();
        switch (style) {
        case CorporateGray:
            CorporateGreyStyle();
            break;
        case CorporateWhite:
            CorporateWhiteStyle();
            break;
        case DarkColors:
            ImGui::StyleColorsDark();
            break;
        case LightColors:
            ImGui::StyleColorsLight();
            break;
        }
        this->styleParam.ResetDirty();
    }

    ImGuiIO& io = ImGui::GetIO();
    this->saveStateDelay += io.DeltaTime;
    if (this->stateParam.IsDirty()) {
        auto state = this->stateParam.Param<core::param::StringParam>()->Value();
        this->windowManager.StateFromJSON(std::string(state));
        this->stateParam.ResetDirty();
    } else if (this->saveState && (this->saveStateDelay > 2.0f)) { // Delayed saving after triggering saving state
        std::string state;
        this->windowManager.StateToJSON(state);
        this->stateParam.Param<core::param::StringParam>()->SetValue(state.c_str(), false);
        this->saveState = false;
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

    if ((instanceTime - this->lastInstanceTime) < 0.0) {
        vislib::sys::Log::DefaultLog.WriteWarn("[GUIView] Current instance time results in negative time delta.");
    }
    io.DeltaTime = ((instanceTime - this->lastInstanceTime) > 0.0)
                       ? (static_cast<float>(instanceTime - this->lastInstanceTime))
                       : (io.DeltaTime);
    this->lastInstanceTime =
        ((instanceTime - this->lastInstanceTime) > 0.0) ? (instanceTime) : (this->lastInstanceTime + io.DeltaTime);

    // Changes that need to be applied before next ImGui::Begin ---------------
    // Loading new font from font window
    if (!this->newFontFilenameToLoad.empty()) {
        ImFontConfig config;
        config.OversampleH = 4;
        config.OversampleV = 1;
        config.GlyphRanges = this->fontUtf8Ranges.data();
        io.Fonts->AddFontFromFileTTF(this->newFontFilenameToLoad.c_str(), this->newFontSizeToLoad, &config);
        ImGui_ImplOpenGL3_CreateFontsTexture();
        /// Load last added font
        io.FontDefault = io.Fonts->Fonts[(io.Fonts->Fonts.Size - 1)];
        this->newFontFilenameToLoad.clear();
    }
    // Loading new font from state
    if (this->newFontIndexToLoad >= 0) {
        if (this->newFontIndexToLoad < io.Fonts->Fonts.Size) {
            io.FontDefault = io.Fonts->Fonts[this->newFontIndexToLoad];
        }
        this->newFontIndexToLoad = -1;
    }
    // Deleting window
    if (!this->windowToDelete.empty()) {
        this->windowManager.DeleteWindowConfiguration(this->windowToDelete);
        this->windowToDelete.clear();
    }

    // Start new frame --------------------------------------------------------
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();

    const auto func = [&, this](const std::string& wn, WindowManager::WindowConfiguration& wc) {
        // Loading font from font window configuration independant if font window is shown
        if (wc.font_reset) {
            if (!wc.font_name.empty()) {
                this->newFontIndexToLoad = -1;
                for (int n = 0; n < io.Fonts->Fonts.Size; n++) {
                    if (std::string(io.Fonts->Fonts[n]->GetDebugName()) == wc.font_name) {
                        this->newFontIndexToLoad = n;
                    }
                }
                if (this->newFontIndexToLoad < 0) {
                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "[GUIView] Could not find font '%s' for loaded state.", wc.font_name.c_str());
                }
            }
            wc.font_reset = false;
        }
        // Draw window content
        if (wc.win_show) {
            ImGui::SetNextWindowBgAlpha(1.0f);
            if (!ImGui::Begin(wn.c_str(), &wc.win_show, wc.win_flags)) {
                ImGui::End(); // early ending
                return;
            }
            // Apply soft reset of window position and size
            if (wc.win_soft_reset) {
                this->windowManager.SoftResetWindowSizePos(wn, wc);
                wc.win_soft_reset = false;
            }
            // Apply reset after new state has been loaded
            if (wc.win_reset) {
                this->windowManager.ResetWindowOnStateLoad(wn, wc);
                wc.win_reset = false;
            }

            // Calling callback drawing window content
            auto cb = this->windowManager.WindowCallback(wc.win_callback);
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
    this->windowManager.EnumWindows(func);

    // Render the frame -------------------------------------------------------
    glViewport(0, 0, viewportWidth, viewportHeight);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return true;
}


void GUIView::drawMainWindowCallback(
    const std::string& window_name, WindowManager::WindowConfiguration& window_config) {
    // Menu -------------------------------------------------------------------
    /// Requires window flag ImGuiWindowFlags_MenuBar
    if (ImGui::BeginMenuBar()) {
        this->drawMenu();
        ImGui::EndMenuBar();
    }

    // Parameters -------------------------------------------------------------
    ImGui::Text("Parameters");
    std::string color_param_help = "[Hover] Parameter for Description Tooltip\n"
                                   "[Right-Click] for Context Menu\n"
                                   "[Drag & Drop] Module Header to other Parameter Window";
    this->popup.HelpMarkerToolTip(color_param_help);

    this->drawParametersCallback(window_name, window_config);
}


void GUIView::drawTFWindowCallback(const std::string& window_name, WindowManager::WindowConfiguration& window_config) {
    this->tfEditor.DrawTransferFunctionEditor();
}


void GUIView::drawParametersCallback(
    const std::string& window_name, WindowManager::WindowConfiguration& window_config) {
    ImGuiStyle& style = ImGui::GetStyle();

    ImGui::PushItemWidth(ImGui::GetContentRegionAvailWidth() * 0.5f); // set general proportional item width

    // Options
    int overrideState = -1; /// invalid
    if (ImGui::Button("Expand All")) {
        overrideState = 1; /// open
    }
    ImGui::SameLine();
    if (ImGui::Button("Collapse All")) {
        overrideState = 0; /// close
    }

    bool show_only_hotkeys = window_config.param_show_hotkeys;
    ImGui::Checkbox("Show Hotkeys", &show_only_hotkeys);
    window_config.param_show_hotkeys = show_only_hotkeys;

    // Offering module filtering only for main parameter view
    if (window_config.win_callback == WindowManager::WindowDrawCallback::MAIN) {
        std::map<int, std::string> opts;
        opts[(int)WindowManager::FilterMode::ALL] = "All";
        opts[(int)WindowManager::FilterMode::INSTANCE] = "Instance";
        opts[(int)WindowManager::FilterMode::VIEW] = "View";
        unsigned int opts_cnt = (unsigned int)opts.size();
        if (ImGui::BeginCombo("Module Filter", opts[(int)window_config.param_module_filter].c_str())) {
            for (unsigned int i = 0; i < opts_cnt; ++i) {

                if (ImGui::Selectable(opts[i].c_str(), ((int)window_config.param_module_filter == i))) {
                    window_config.param_module_filter = (WindowManager::FilterMode)i;
                    window_config.param_modules_list.clear();
                    if ((window_config.param_module_filter == WindowManager::FilterMode::INSTANCE) ||
                        (window_config.param_module_filter == WindowManager::FilterMode::VIEW)) {

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
                            if (window_config.param_module_filter == WindowManager::FilterMode::INSTANCE) {
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
                                            window_config.param_modules_list.emplace_back(modname);
                                        }
                                    };
                                    this->GetCoreInstance()->EnumModulesNoLock(nullptr, func);
                                }
                            } else { // (window_config.param_module_filter == WindowManager::FilterMode::VIEW)
                                // Considering modules depending on their connection to the first VIEW this module is
                                // connected to.
                                const auto add_func = [&, this](core::Module* mod) {
                                    std::string modname = mod->FullName().PeekBuffer();
                                    window_config.param_modules_list.emplace_back(modname);
                                };
                                this->GetCoreInstance()->EnumModulesNoLock(viewname, add_func);
                            }
                        } else {
                            vislib::sys::Log::DefaultLog.WriteWarn("[GUIView] Could not find abstract view "
                                                                   "module this gui is connected to.");
                        }
                    }
                }
                std::string hover = "Show all Modules."; // == WindowManager::FilterMode::ALL
                if (i == (int)WindowManager::FilterMode::INSTANCE) {
                    hover = "Show Modules with same Instance Name as current View and Modules with no Instance Name.";
                } else if (i == (int)WindowManager::FilterMode::VIEW) {
                    hover = "Show Modules subsequently connected to the View Module the Gui Module is connected to.";
                }
                this->popup.HoverToolTip(hover);
            }
            ImGui::EndCombo();
        }
        this->popup.HelpMarkerToolTip("Filter applies globally to all parameter windows.\n"
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
            if (!this->considerModule(label, window_config.param_modules_list)) {
                current_mod_open = false;
                return;
            }

            // Main parameter window always draws all module's parameters
            if (window_config.win_callback != WindowManager::WindowDrawCallback::MAIN) {
                // Consider only modules contained in list
                if (std::find(window_config.param_modules_list.begin(), window_config.param_modules_list.end(),
                        label) == window_config.param_modules_list.end()) {
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

            // Context menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Copy to new Window")) {
                    // using instance time as hidden unique id
                    std::string window_name = "Parameters###parameters" + std::to_string(this->lastInstanceTime);

                    WindowManager::WindowConfiguration tmp_win;
                    tmp_win.win_show = true;
                    tmp_win.win_flags = ImGuiWindowFlags_HorizontalScrollbar;
                    tmp_win.win_callback = WindowManager::WindowDrawCallback::PARAM;
                    tmp_win.param_show_hotkeys = false;
                    tmp_win.param_modules_list.emplace_back(label);
                    this->windowManager.AddWindowConfiguration(window_name, tmp_win);
                }
                // Deleting module's parameters is not available in main parameter window.
                if (window_config.win_callback != WindowManager::WindowDrawCallback::MAIN) {
                    if (ImGui::MenuItem("Delete from List")) {
                        std::vector<std::string>::iterator find_iter = std::find(
                            window_config.param_modules_list.begin(), window_config.param_modules_list.end(), label);
                        // Break if module name is not contained in list
                        if (find_iter != window_config.param_modules_list.end()) {
                            window_config.param_modules_list.erase(find_iter);
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
                if (window_config.param_show_hotkeys) {
                    this->drawParameterHotkey(mod, slot);
                } else {
                    this->drawParameter(mod, slot);
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
    ImGui::InvisibleButton("Drop Area", ImVec2(ImGui::GetContentRegionAvailWidth(), ImGui::GetFontSize()));
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("DND_COPY_MODULE_PARAMETERS")) {

            IM_ASSERT(payload->DataSize == (dnd_size * sizeof(char)));
            std::string payload_id = (const char*)payload->Data;

            // Nothing to add to main parameter window (draws always all module's parameters)
            if ((window_config.win_callback != WindowManager::WindowDrawCallback::MAIN)) {
                // Insert dragged module name only if not contained in list
                if (std::find(window_config.param_modules_list.begin(), window_config.param_modules_list.end(),
                        payload_id) == window_config.param_modules_list.end()) {
                    window_config.param_modules_list.emplace_back(payload_id);
                }
            }
        }
        ImGui::EndDragDropTarget();
    }

    ImGui::PopItemWidth();
}


void GUIView::drawFpsWindowCallback(const std::string& window_name, WindowManager::WindowConfiguration& window_config) {
    ImGuiIO& io = ImGui::GetIO();

    window_config.fpsms_current_delay += io.DeltaTime;
    if (window_config.fpsms_max_delay <= 0.0f) {
        return;
    }
    if (window_config.fpsms_max_value_count == 0) {
        window_config.fpsms_fps_values.clear();
        window_config.fpsms_ms_values.clear();
        return;
    }

    if (window_config.fpsms_current_delay > (1.0f / window_config.fpsms_max_delay)) {
        // Leave some space in histogram for text of current value
        const float scale_fac = 1.5f;

        if (window_config.fpsms_fps_values.size() != window_config.fpsms_ms_values.size()) {
            vislib::sys::Log::DefaultLog.WriteError(
                "[GUIView] Arrays for FPS and frame times do not have equal length.");
            return;
        }

        int size = (int)window_config.fpsms_fps_values.size();
        if (size != window_config.fpsms_max_value_count) {
            if (size > window_config.fpsms_max_value_count) {
                window_config.fpsms_fps_values.erase(window_config.fpsms_fps_values.begin(),
                    window_config.fpsms_fps_values.begin() + (size - window_config.fpsms_max_value_count));
                window_config.fpsms_ms_values.erase(window_config.fpsms_ms_values.begin(),
                    window_config.fpsms_ms_values.begin() + (size - window_config.fpsms_max_value_count));

            } else if (size < window_config.fpsms_max_value_count) {
                window_config.fpsms_fps_values.insert(
                    window_config.fpsms_fps_values.begin(), (window_config.fpsms_max_value_count - size), 0.0f);
                window_config.fpsms_ms_values.insert(
                    window_config.fpsms_ms_values.begin(), (window_config.fpsms_max_value_count - size), 0.0f);
            }
        }
        if (size > 0) {
            window_config.fpsms_fps_values.erase(window_config.fpsms_fps_values.begin());
            window_config.fpsms_ms_values.erase(window_config.fpsms_ms_values.begin());

            window_config.fpsms_fps_values.emplace_back(io.Framerate);
            window_config.fpsms_ms_values.emplace_back(io.DeltaTime * 1000.0f); // scale to milliseconds

            float value_max = 0.0f;
            for (auto& v : window_config.fpsms_fps_values) {
                value_max = (v > value_max) ? (v) : (value_max);
            }
            window_config.fpsms_fps_value_scale = value_max * scale_fac;

            value_max = 0.0f;
            for (auto& v : window_config.fpsms_ms_values) {
                value_max = (v > value_max) ? (v) : (value_max);
            }
            window_config.fpsms_ms_value_scale = value_max * scale_fac;
        }

        window_config.fpsms_current_delay = 0.0f;
    }

    // Draw window content
    if (ImGui::RadioButton("fps", (window_config.fpsms_mode == WindowManager::TimingMode::FPS))) {
        window_config.fpsms_mode = WindowManager::TimingMode::FPS;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("ms", (window_config.fpsms_mode == WindowManager::TimingMode::MS))) {
        window_config.fpsms_mode = WindowManager::TimingMode::MS;
    }

    ImGui::SameLine(0.0f, 50.0f);
    ImGui::Checkbox("Options", &window_config.fpsms_show_options);

    // Default for window_config.fpsms_mode == WindowManager::TimingMode::FPS
    std::vector<float>* arr = &window_config.fpsms_fps_values;
    float val_scale = window_config.fpsms_fps_value_scale;
    if (window_config.fpsms_mode == WindowManager::TimingMode::MS) {
        arr = &window_config.fpsms_ms_values;
        val_scale = window_config.fpsms_ms_value_scale;
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

    if (window_config.fpsms_show_options) {
        float rate = window_config.fpsms_max_delay;
        if (ImGui::InputFloat("Refresh Rate", &rate, 0.0f, 0.0f, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue)) {
            // Validate refresh rate
            window_config.fpsms_max_delay = std::max(0.0f, rate);
            window_config.fpsms_fps_values.clear();
            window_config.fpsms_ms_values.clear();
        }
        std::string help = "Changes clear all values";
        this->popup.HelpMarkerToolTip(help);

        int mvc = window_config.fpsms_max_value_count;
        if (ImGui::InputInt("History Size", &mvc, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue)) {
            // Validate refresh rate
            window_config.fpsms_max_value_count = std::max(0, mvc);
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

        ImGui::Text("Copy to Clipborad");
        help = "Values are copied in chronological order (newest first)";
        this->popup.HelpMarkerToolTip(help);
    }
}


void GUIView::drawFontWindowCallback(
    const std::string& window_name, WindowManager::WindowConfiguration& window_config) {
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
    window_config.font_name = std::string(font_current->GetDebugName());

#ifdef GUI_USE_FILEUTILS
    ImGui::Separator();
    ImGui::Text("Load new Font from File");

    std::string label = "Font Filename (.ttf)";
    vislib::StringA valueString;
    vislib::UTF8Encoder::Encode(valueString, vislib::StringA(window_config.font_new_filename.c_str()));
    std::string valueUtf8String(valueString.PeekBuffer());
    ImGuiInputTextFlags textflags = ImGuiInputTextFlags_AutoSelectAll;
    ImGui::InputText(label.c_str(), &valueUtf8String, textflags);
    vislib::UTF8Encoder::Decode(valueString, vislib::StringA(valueUtf8String.data()));
    window_config.font_new_filename = valueString.PeekBuffer();

    label = "Font Size";
    ImGui::InputFloat(label.c_str(), &window_config.font_new_size, 0.0f, 0.0f, "%.2f", ImGuiInputTextFlags_None);
    // Validate font size
    if (window_config.font_new_size <= 0.0f) {
        window_config.font_new_size = 5.0f; /// min valid font size
    }

    // Validate font file before offering load button
    if (HasExistingFileExtension(window_config.font_new_filename, std::string(".ttf"))) {
        if (ImGui::Button("Add Font")) {
            this->newFontFilenameToLoad = window_config.font_new_filename;
            this->newFontSizeToLoad = window_config.font_new_size;
        }
    } else {
        ImGui::TextColored(ImVec4(0.9f, 0.0f, 0.0f, 1.0f), "Please enter valid font file name");
    }
    std::string help = "Same font can be loaded multiple times using different font size";
    this->popup.HelpMarkerToolTip(help);
#endif // GUI_USE_FILEUTILS
}


void GUIView::drawMenu(void) {

    bool open_popup_project = false;
    if (ImGui::BeginMenu("File")) {
#ifdef GUI_USE_FILEUTILS
        // Load/save parameter values to LUA file
        if (ImGui::MenuItem("Save Project")) {
            open_popup_project = true;
        }
        // if (ImGui::MenuItem("Load Project")) {
        //    // TODO:  Load parameter file
        //    std::string projectFilename;
        //    this->GetCoreInstance()->LoadProject(vislib::StringA(projectFilename.c_str()));
        //}
        ImGui::Separator();
#endif // GUI_USE_FILEUTILS
        if (ImGui::MenuItem("Exit", "'Esc', ALT + 'F4'")) {
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
                        this->windowToDelete = wn;
                    }
                    ImGui::EndPopup();
                }
                this->popup.HoverToolTip("[Right-Click] to open Context Menu for Deleting Window Permanently.");
            } else {
                this->popup.HoverToolTip(
                    "['Window Hotkey'] to Show/Hide Window.\n[Shift]+['Window Hotkey'] to Reset Size "
                    "and Position of Window.");
            }
        };
        this->windowManager.EnumWindows(func);

        ImGui::EndMenu();
    }

    // Help
    bool open_popup_about = false;
    if (ImGui::BeginMenu("Help")) {
        const std::string gitLink = "https://github.com/UniStuttgart-VISUS/megamol";
        const std::string webLink = "https://megamol.org/";
        const std::string hint = "Copy Link to Clipboard";
        if (ImGui::MenuItem("Website")) {
            ImGui::SetClipboardText(webLink.c_str());
        }
        this->popup.HoverToolTip(hint);
        if (ImGui::MenuItem("GitHub")) {
            ImGui::SetClipboardText(gitLink.c_str());
        }
        this->popup.HoverToolTip(hint);
        ImGui::Separator();
        if (ImGui::MenuItem("About...")) {
            open_popup_about = true;
        }
        ImGui::EndMenu();
    }

    // Popups
    if (open_popup_about) {
        ImGui::OpenPopup("About");
    }
    if (ImGui::BeginPopupModal("About", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {

        std::string about =
            std::string("MegaMol - Version ") + std::to_string(MEGAMOL_CORE_MAJOR_VER) + (".") +
            std::to_string(MEGAMOL_CORE_MINOR_VER) + ("\n(git hash ") + std::string(MEGAMOL_CORE_COMP_REV) +
            (")\n\nDear ImGui - Version ") + std::string(IMGUI_VERSION) +
            std::string("\n\nContact: megamol@visus.uni-stuttgart.de\nWeb: https://megamol.org\nGit-Hub: "
                        "https://github.com/UniStuttgart-VISUS/megamol\n\nCopyright (C) 2009-2019 by Universitaet "
                        "Stuttgart (VIS).\nAll rights "
                        "reserved.");

        ImGui::Text(about.c_str());
        ImGui::Separator();
        if (ImGui::Button("Close")) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::SetItemDefaultFocus();
        ImGui::EndPopup();
    }

#ifdef GUI_USE_FILEUTILS
    if (open_popup_project) {
        ImGui::OpenPopup("Save Project");
    }
    if (ImGui::BeginPopupModal("Save Project", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {


        std::string label = "File Name";
        vislib::StringA valueString;
        vislib::UTF8Encoder::Encode(valueString, vislib::StringA(this->projectFilename.c_str()));
        std::string valueUtf8String(valueString.PeekBuffer());
        ImGuiInputTextFlags textflags = ImGuiInputTextFlags_AutoSelectAll;
        ImGui::InputText(label.c_str(), &valueUtf8String, textflags);
        vislib::UTF8Encoder::Decode(valueString, vislib::StringA(valueUtf8String.data()));
        this->projectFilename = valueString.PeekBuffer();

        bool valid = false;
        if (!HasFileExtension(this->projectFilename, std::string(".lua"))) {
            ImGui::TextColored(ImVec4(0.9f, 0.0f, 0.0f, 1.0f), "File name needs to have the ending '.lua'");
        } else {
            valid = true;
        }
        if (PathExists(this->projectFilename)) {
            ImGui::TextColored(ImVec4(0.9f, 0.0f, 0.0f, 1.0f), "File name already exists and will be overwritten!");
        }

        if (ImGui::Button("Save")) {
            if (valid) {
                if (SaveProjectFile(this->projectFilename, this->GetCoreInstance())) {
                    ImGui::CloseCurrentPopup();
                }
            }
        }
        ImGui::SetItemDefaultFocus();
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
    if (!param.IsNull() && param->IsGUIVisible()) {
        // Set different style if parameter is read-only
        if (param->IsGUIReadOnly()) {
            ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.25f);
        }

        std::string param_name = slot.Name().PeekBuffer();
        std::string param_id = std::string(mod.FullName().PeekBuffer()) + "::" + param_name;
        auto pos = param_name.find("::");
        if (pos != std::string::npos) {
            param_name = param_name.substr(pos + 2);
        }
        std::string param_label = param_name + "###" + param_id;
        std::string param_desc = slot.Description().PeekBuffer();
        std::string float_format = "%.7f";

        if (auto* p = slot.template Param<core::param::BoolParam>()) {
            auto value = p->Value();
            if (ImGui::Checkbox(param_label.c_str(), &value)) {
                p->SetValue(value);
            }
        } else if (auto* p = slot.template Param<core::param::ButtonParam>()) {
            std::string hotkey = " (";
            hotkey.append(p->GetKeyCode().ToString());
            hotkey.append(")");
            // no check if found -> should be present
            auto insert_pos = param_label.find("###");
            param_label.insert(insert_pos, hotkey);

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

            ImGui::Separator();
            ImGui::Text(param_name.c_str());

            bool load_tf = false;
            param_label = "Load into Editor###editor" + param_id;
            if (p == this->tfEditor.GetActiveParameter()) {
                param_label = "Open Editor###editor" + param_id;
            }
            if (ImGui::Button(param_label.c_str())) {
                this->tfEditor.SetActiveParameter(p);
                load_tf = true;
                // Open window calling the transfer function editor callback
                const auto func = [&, this](const std::string& wn, WindowManager::WindowConfiguration& wc) {
                    if (wc.win_callback == WindowManager::WindowDrawCallback::TF) {
                        wc.win_show = true;
                    }
                };
                this->windowManager.EnumWindows(func);
            }
            ImGui::SameLine();
            param_label = "Copy to Clipboard###clipboard" + param_id;
            if (ImGui::Button(param_label.c_str())) {
                ImGui::SetClipboardText(p->Value().c_str());
            }
            ImGui::SameLine();
            param_label = "Copy from Clipboard###fclipboard" + param_id;
            if (ImGui::Button(param_label.c_str())) {
                p->SetValue(ImGui::GetClipboardText());
                load_tf = true;
            }
            ImGui::SameLine();
            param_label = "Reset###reset" + param_id;
            if (ImGui::Button(param_label.c_str())) {
                p->SetValue("");
                load_tf = true;
            }

            if (p == this->tfEditor.GetActiveParameter()) {
                ImGui::TextColored(style.Colors[ImGuiCol_ButtonActive], "Currently loaded into Editor");
            }

            ImGui::PushTextWrapPos(ImGui::GetContentRegionAvailWidth());
            ImGui::Text("JSON: ");
            ImGui::SameLine();
            ImGui::TextDisabled(p->Value().c_str());
            ImGui::PopTextWrapPos();

            // Loading new transfer function string from parameter into editor
            if (load_tf && (p == this->tfEditor.GetActiveParameter())) {
                if (!this->tfEditor.SetTransferFunction(p->Value())) {
                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "[GUIView] Could not load transfer function of parameter: %s.", param_id.c_str());
                }
            }

            ImGui::Separator();
        } else if (auto* p = slot.template Param<core::param::EnumParam>()) {
            // XXX: no UTF8 fanciness required here?
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
            // XXX: no UTF8 fanciness required here?
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
            auto value = p->Value();
            if (ImGui::InputFloat(param_label.c_str(), &value, 0.0f, 0.0f, float_format.c_str(),
                    ImGuiInputTextFlags_EnterReturnsTrue)) {
                p->SetValue(std::max(p->MinValue(), std::min(value, p->MaxValue())));
            }
        } else if (auto* p = slot.template Param<core::param::IntParam>()) {
            auto value = p->Value();
            if (ImGui::InputInt(param_label.c_str(), &value, 0, 0, ImGuiInputTextFlags_EnterReturnsTrue)) {
                p->SetValue(std::max(p->MinValue(), std::min(value, p->MaxValue())));
            }
        } else if (auto* p = slot.template Param<core::param::Vector2fParam>()) {
            auto value = p->Value();
            if (ImGui::InputFloat2(param_label.c_str(), value.PeekComponents(), float_format.c_str(),
                    ImGuiInputTextFlags_EnterReturnsTrue)) {
                p->SetValue(value);
            }
        } else if (auto* p = slot.template Param<core::param::Vector3fParam>()) {
            auto value = p->Value();
            if (ImGui::InputFloat3(param_label.c_str(), value.PeekComponents(), float_format.c_str(),
                    ImGuiInputTextFlags_EnterReturnsTrue)) {
                p->SetValue(value);
            }
        } else if (auto* p = slot.template Param<core::param::Vector4fParam>()) {
            auto value = p->Value();
            if (ImGui::InputFloat4(param_label.c_str(), value.PeekComponents(), float_format.c_str(),
                    ImGuiInputTextFlags_EnterReturnsTrue)) {
                p->SetValue(value);
            }
        } else { // if (auto* p = slot.Param<core::param::StringParam>()) {
            // XXX: UTF8 conversion and allocation every frame is horrific inefficient.
            vislib::StringA valueString;
            vislib::UTF8Encoder::Encode(valueString, param->ValueString());
            std::string valueUtf8String(valueString.PeekBuffer());

            ImGuiInputTextFlags textflags = ImGuiInputTextFlags_CtrlEnterForNewLine |
                                            ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll;

            // Determine line count
            int nlcnt = 0;
            for (auto& c : valueUtf8String) {
                if (c == '\n') {
                    nlcnt++;
                }
            }
            nlcnt = std::min(5, nlcnt);
            if (nlcnt > 0) {

                ImVec2 ml_dim =
                    ImVec2(ImGui::CalcItemWidth(), ImGui::GetFrameHeight() + (ImGui::GetFontSize() * (float)(nlcnt)));

                if (ImGui::InputTextMultiline(param_label.c_str(), &valueUtf8String, ml_dim, textflags)) {
                    vislib::UTF8Encoder::Decode(valueString, vislib::StringA(valueUtf8String.data()));
                    param->ParseValue(valueString);
                }
            } else {
                if (ImGui::InputText(param_label.c_str(), &valueUtf8String, textflags)) {
                    vislib::UTF8Encoder::Decode(valueString, vislib::StringA(valueUtf8String.data()));
                    param->ParseValue(valueString);
                }
                help = "[Ctrl + Enter] for new line.\nPress [Return] to confirm changes.";
            }
        }

        this->popup.HoverToolTip(param_desc, ImGui::GetID(param_label.c_str()), 1.0f);

        this->popup.HelpMarkerToolTip(help);

        // Reset to default style
        if (param->IsGUIReadOnly()) {
            ImGui::PopItemFlag();
            ImGui::PopStyleVar();
        }
    }
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
            this->popup.HoverToolTip(desc);

            ImGui::NextColumn();

            ImGui::Text(keycode.c_str());
            this->popup.HoverToolTip(desc);

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
    if (this->checkHotkeysOnce) {

        std::list<core::view::KeyCode> hotkeylist;
        hotkeylist.clear();

        this->GetCoreInstance()->EnumParameters([&, this](const auto& mod, auto& slot) {
            auto param = slot.Parameter();
            if (!param.IsNull()) {
                if (auto* p = slot.template Param<core::param::ButtonParam>()) {
                    auto hotkey = p->GetKeyCode();

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

        this->checkHotkeysOnce = false;
    }
}


void GUIView::shutdown(void) {
    vislib::sys::Log::DefaultLog.WriteInfo("[GUIView] Triggering MegaMol instance shutdown...");
    this->GetCoreInstance()->Shutdown();
}
