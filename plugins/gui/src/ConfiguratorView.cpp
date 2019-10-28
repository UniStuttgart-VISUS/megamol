/*
 * ConfiguratorView.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ConfiguratorView.h"

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
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"

#include "CorporateGreyStyle.h"
#include "CorporateWhiteStyle.h"


using namespace megamol;
using namespace megamol::gui;
using vislib::sys::Log;


ConfiguratorView::ConfiguratorView()
    : core::view::AbstractView()
    , render_view_slot("renderview", "Connects to a preceding RenderView that will be decorated with a Configurator")
    , style_param("style", "Color style, theme")
    , context(nullptr)
    , utils()
    , font_utf8_ranges()
    , last_instance_time(0.0)
 {

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

}

ConfiguratorView::~ConfiguratorView() { this->Release(); }

bool ConfiguratorView::create() {
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
        vislib::sys::Log::DefaultLog.WriteError("[ConfiguratorView] Could not create ImGui context");
        return false;
    }
    ImGui::SetCurrentContext(this->context);

    // Init OpenGL for ImGui --------------------------------------------------
    const char* glsl_version = "#version 130"; /// "#version 150"
    ImGui_ImplOpenGL3_Init(glsl_version);

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
    this->font_utf8_ranges.clear();
    this->font_utf8_ranges.emplace_back(0x0020);
    this->font_utf8_ranges.emplace_back(0x03FF); // Basic Latin + Latin Supplement + Greek Alphabet
    this->font_utf8_ranges.emplace_back(0x20AC);
    this->font_utf8_ranges.emplace_back(0x20AC); // Euro
    this->font_utf8_ranges.emplace_back(0x2122);
    this->font_utf8_ranges.emplace_back(0x2122); // TM
    this->font_utf8_ranges.emplace_back(0x212B);
    this->font_utf8_ranges.emplace_back(0x212B); // Angstroem
    this->font_utf8_ranges.emplace_back(0x0391);
    this->font_utf8_ranges.emplace_back(0); // (range termination)

    // Load initial fonts only once for all imgui contexts --------------------
    if (!other_context) {
        ImFontConfig config;
        config.OversampleH = 6;
        config.GlyphRanges = this->font_utf8_ranges.data();
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


void ConfiguratorView::release() {
    if (this->context != nullptr) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui::DestroyContext(this->context);
    }
}


void ConfiguratorView::unpackMouseCoordinates(float& x, float& y) {
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


float ConfiguratorView::DefaultTime(double instTime) const {
    // This view does not do any time control
    return 0.0f;
}


unsigned int ConfiguratorView::GetCameraSyncNumber(void) const {
    Log::DefaultLog.WriteWarn("ConfiguratorView::GetCameraSyncNumber unsupported");
    return 0u;
}


void ConfiguratorView::SerialiseCamera(vislib::Serialiser& serialiser) const {
    Log::DefaultLog.WriteWarn("ConfiguratorView::SerialiseCamera unsupported");
}


void ConfiguratorView::DeserialiseCamera(vislib::Serialiser& serialiser) {
    Log::DefaultLog.WriteWarn("ConfiguratorView::DeserialiseCamera unsupported");
}


void ConfiguratorView::Render(const mmcRenderViewContext& context) {
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
        this->drawConfigurator(crv->GetViewport(), crv->InstanceTime());
    } else {
        ::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        ::glClear(GL_COLOR_BUFFER_BIT);
        if (this->overrideCall != nullptr) {
            this->drawConfigurator(this->overrideCall->GetViewport(), context.InstanceTime);
        }
    }
    if (this->doHookCode()) {
        this->doAfterRenderHook();
    }
}


void ConfiguratorView::ResetView(void) {
    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
    if (crv) {
        (*crv)(core::view::CallRenderView::CALL_RESETVIEW);
    }
}


void ConfiguratorView::Resize(unsigned int width, unsigned int height) {
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


void ConfiguratorView::UpdateFreeze(bool freeze) {
    auto* crv = this->render_view_slot.CallAs<core::view::CallRenderView>();
    if (crv) {
        auto callType = freeze ? core::view::CallRenderView::CALL_FREEZE : core::view::CallRenderView::CALL_UNFREEZE;
        (*crv)(callType);
    }
}


bool ConfiguratorView::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {
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

    // HotKeys ----------------------------------------------------------------
    // NB: Hotkey processing is stopped after first occurence. Order of hotkey processing is crucial.
    // Hotkeys always trigger just one event.

    ///EMPTY///

    // ------------------------------------------------------------------------

    if (hotkeyPressed) return true;

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


bool ConfiguratorView::OnChar(unsigned int codePoint) {
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


bool ConfiguratorView::OnMouseMove(double x, double y) {
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


bool ConfiguratorView::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {

    ImGui::SetCurrentContext(this->context);

    bool down = (action == core::view::MouseButtonAction::PRESS);
    auto buttonIndex = static_cast<size_t>(button);
    ImGuiIO& io = ImGui::GetIO();

    auto hoverFlags = ImGuiHoveredFlags_AnyWindow | ImGuiHoveredFlags_AllowWhenDisabled |
                      ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem;

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


bool ConfiguratorView::OnMouseScroll(double dx, double dy) {
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


bool ConfiguratorView::OnRenderView(megamol::core::Call& call) {
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


void ConfiguratorView::validateConfigurator(void) {
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

}


bool ConfiguratorView::drawConfigurator(vislib::math::Rectangle<int> viewport, double instanceTime) {
    ImGui::SetCurrentContext(this->context);

    auto viewportWidth = viewport.Width();
    auto viewportHeight = viewport.Height();

    // Set IO stuff
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)viewportWidth, (float)viewportHeight);
    io.DisplayFramebufferScale = ImVec2(1.0, 1.0);

    if ((instanceTime - this->last_instance_time) < 0.0) {
        vislib::sys::Log::DefaultLog.WriteWarn("[ConfiguratorView] Current instance time results in negative time delta.");
    }
    io.DeltaTime = ((instanceTime - this->last_instance_time) > 0.0)
                       ? (static_cast<float>(instanceTime - this->last_instance_time))
                       : (io.DeltaTime);
    this->last_instance_time = ((instanceTime - this->last_instance_time) > 0.0)
                                         ? (instanceTime)
                                         : (this->last_instance_time + io.DeltaTime);

    // ... --------------------------------------------------------------------
   
    ///EMPTY///












    // Render current frame ---------------------------------------------------

    glViewport(0, 0, viewportWidth, viewportHeight);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return true;
}

