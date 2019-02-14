/*
 * GUIRenderer.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GUIRENDERER_H_INCLUDED
#define MEGAMOL_GUI_GUIRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <iomanip> // setprecision
#include <sstream> // stringstream

#include "mmcore/CallerSlot.h"
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
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmcore/view/Renderer3DModule.h"

#include "vislib/UTF8Encoder.h"

#include <imgui.h>
#include "imgui_impl_opengl3.h"


namespace megamol {
namespace gui {

template <class M, class C> class GUIRenderer : public M {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void);

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description(void) { return "Graphical user interface renderer"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable(void) { return true; }

    /**
     * Initialises a new instance.
     */
    GUIRenderer();

    /**
     * Finalises an instance.
     */
    virtual ~GUIRenderer();

protected:
    virtual bool create() override;

    virtual void release() override;

    virtual bool OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) override;

    virtual bool OnChar(unsigned int codePoint) override;

    virtual bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

    virtual bool OnMouseMove(double x, double y) override;

    virtual bool OnMouseScroll(double dx, double dy) override;

    virtual bool GetExtents(C& call) override;

    virtual bool Render(C& call) override;


private:
    /**
     * Draws the main menu bar.
     */
    void drawMainMenu(void);

    /**
     * Draws the menu bar.
     */
    void drawMenu(void);

    /**
     * Draws the parameter window.
     */
    void drawParameterWindow(void);

    /**
     * Draws a parameter for the parameter window.
     */
    void drawParameter(const core::Module& mod, core::param::ParamSlot& slot);

    /**
     * Map vislib::sys::KeyCode to megamol::core::view::Key as integer
     */
    int keyCode2Key(vislib::sys::KeyCode keycode);

    /** The decorated renderer caller slot */
    core::CallerSlot decoratedRendererSlot;

    // Global ImGui Stata Variables  ------------------------------------------

    bool parameterWindowOpen;
    float fpsDelay;
    std::string fps;


    // ------------------------------------------------------------------------
};


typedef GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D> GUIRenderer2D;
typedef GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D> GUIRenderer3D;


template <>
inline GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::GUIRenderer()
    : decoratedRendererSlot("decoratedRenderer", "Connects to another 2D Renderer being decorated") {

    this->decoratedRendererSlot.SetCompatibleCall<core::view::CallRender2DDescription>();
    this->MakeSlotAvailable(&this->decoratedRendererSlot);
}


template <>
inline GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::GUIRenderer()
    : decoratedRendererSlot("decoratedRenderer", "Connects to another 3D Renderer being decorated") {

    this->decoratedRendererSlot.SetCompatibleCall<core::view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->decoratedRendererSlot);
}


template <> inline const char* GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::ClassName(void) {

    return "GUIRenderer2D";
}


template <> inline const char* GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::ClassName(void) {

    return "GUIRenderer3D";
}


template <class M, class C> GUIRenderer<M, C>::~GUIRenderer() { this->Release(); }


template <class M, class C> bool GUIRenderer<M, C>::create() {

    // Create ImGui context ---------------------------------------------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // IO settings ------------------------------------------------------------
    ImGuiIO& io = ImGui::GetIO();

    // ImGui Key Map
    io.KeyMap[ImGuiKey_::ImGuiKey_Tab] = static_cast<int>(core::view::Key::KEY_TAB);
    io.KeyMap[ImGuiKey_::ImGuiKey_LeftArrow] = static_cast<int>(core::view::Key::KEY_LEFT);
    io.KeyMap[ImGuiKey_::ImGuiKey_RightArrow] = static_cast<int>(core::view::Key::KEY_RIGHT);
    io.KeyMap[ImGuiKey_::ImGuiKey_UpArrow] = static_cast<int>(core::view::Key::KEY_UP);
    io.KeyMap[ImGuiKey_::ImGuiKey_DownArrow] = static_cast<int>(core::view::Key::KEY_DOWN);
    io.KeyMap[ImGuiKey_::ImGuiKey_PageUp] = static_cast<int>(core::view::Key::KEY_PAGE_UP);
    io.KeyMap[ImGuiKey_::ImGuiKey_PageDown] = static_cast<int>(core::view::Key::KEY_PAGE_DOWN);
    io.KeyMap[ImGuiKey_::ImGuiKey_Home] = static_cast<int>(core::view::Key::KEY_HOME);
    io.KeyMap[ImGuiKey_::ImGuiKey_End] = static_cast<int>(core::view::Key::KEY_END);
    io.KeyMap[ImGuiKey_::ImGuiKey_Insert] = static_cast<int>(core::view::Key::KEY_INSERT);
    io.KeyMap[ImGuiKey_::ImGuiKey_Delete] = static_cast<int>(core::view::Key::KEY_DELETE);
    io.KeyMap[ImGuiKey_::ImGuiKey_Backspace] = static_cast<int>(core::view::Key::KEY_BACKSPACE);
    io.KeyMap[ImGuiKey_::ImGuiKey_Space] = static_cast<int>(core::view::Key::KEY_SPACE);
    io.KeyMap[ImGuiKey_::ImGuiKey_Enter] = static_cast<int>(core::view::Key::KEY_ENTER);
    io.KeyMap[ImGuiKey_::ImGuiKey_Escape] = static_cast<int>(core::view::Key::KEY_ESCAPE);
    // io.KeyMap[ImGuiKey_::ImGuiKey_A] = -1;
    // io.KeyMap[ImGuiKey_::ImGuiKey_C] = -1;
    // io.KeyMap[ImGuiKey_::ImGuiKey_V] = -1;
    // io.KeyMap[ImGuiKey_::ImGuiKey_X] = -1;
    // io.KeyMap[ImGuiKey_::ImGuiKey_Y] = -1;
    // io.KeyMap[ImGuiKey_::ImGuiKey_Z] = -1;

    // Style settings ---------------------------------------------------------
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.FrameRounding = 5.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.AntiAliasedLines = true;
    style.AntiAliasedFill = true;

    // Init OpenGL for ImGui --------------------------------------------------
    const char* glsl_version = "#version 150";
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Init state variables ---------------------------------------------------
    this->parameterWindowOpen = true;
    this->fpsDelay = 1.0f;
    this->fps = "";

    return true;
}


template <class M, class C> void GUIRenderer<M, C>::release() {

    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();
}


template <>
inline bool GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::GetExtents(
    core::view::CallRender2D& call) {

    auto* cr = this->decoratedRendererSlot.CallAs<core::view::CallRender2D>();
    if (cr != NULL) {
        (*cr) = call;
        if ((*cr)(core::view::AbstractCallRender::FnGetExtents)) {
            call = (*cr);
        }
    } else {
        call.SetBoundingBox(vislib::math::Rectangle<float>(0, 1, 1, 0));
    }

    return true;
}


template <>
inline bool GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::GetExtents(
    core::view::CallRender3D& call) {

    auto* cr = this->decoratedRendererSlot.CallAs<core::view::CallRender3D>();
    if (cr != NULL) {
        (*cr) = call;
        if ((*cr)(core::view::AbstractCallRender::FnGetExtents)) {
            call = (*cr);
        }
    } else {
        call.AccessBoundingBoxes().Clear();
        call.AccessBoundingBoxes().SetWorldSpaceBBox(
            vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f));
    }

    return true;
}


template <class M, class C> bool GUIRenderer<M, C>::Render(C& call) {

    auto* cr = this->decoratedRendererSlot.template CallAs<C>();
    if (cr != NULL) {
        (*cr) = call;
        if ((*cr)(core::view::AbstractCallRender::FnRender)) {
            call = (*cr);
        }
    }

    auto viewportWidth = call.GetViewport().Width();
    auto viewportHeight = call.GetViewport().Height();

    // Set IO stuff
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(viewportWidth, viewportHeight);
    io.DisplayFramebufferScale = ImVec2(1.0, 1.0);
    io.DeltaTime = static_cast<float>(call.LastFrameTime() / 1000.0); // in milliseconds

    // Start the frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();

    // Construct frame
    // this->drawMainMenu();
    // ImGui::ShowMetricsWindow(nullptr);
    this->drawParameterWindow();

    // Render the frame
    glViewport(0, 0, viewportWidth, viewportHeight);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return true;
}


template <class M, class C> void GUIRenderer<M, C>::drawMainMenu(void) {

    if (ImGui::BeginMainMenuBar()) {
        this->drawMenu();
        ImGui::EndMainMenuBar();
    }
}


template <class M, class C> void GUIRenderer<M, C>::drawParameterWindow(void) {

    // Window -----------------------------------------------------------------
    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_HorizontalScrollbar;
    bool* p_open = nullptr; // &this->parameterWindowOpen;
    std::stringstream stream;
    stream << "MegaMol (Dear ImGUI" << IMGUI_VERSION << ")";
    std::string title = stream.str();
    ImGui::Begin(title.data(), p_open, window_flags);

    // Menu -------------------------------------------------------------------
    if (ImGui::BeginMenuBar()) {
        this->drawMenu();
        ImGui::EndMenuBar();
    }

    // Parameters -------------------------------------------------------------
    ImGui::Text("Parameters: ");
    const core::Module* currentMod = nullptr;
    bool currentModOpen = false;
    this->GetCoreInstance()->EnumParameters([&, this](const auto& mod, auto& slot) {
        if (currentMod != &mod) {
            currentMod = &mod;
            // Set to "open" by default.
            // auto headerId = ImGui::GetID(mod.FullName());
            // int headerState = ImGui::GetStateStorage()->GetInt(headerId, 1); // 0=close 1=open
            // ImGui::GetStateStorage()->SetInt(headerId, headerState);

            currentModOpen = ImGui::CollapsingHeader(mod.FullName());
        }
        if (currentModOpen) {
            this->drawParameter(mod, slot);
        }
    });

    ImGui::End();
}


template <class M, class C> void GUIRenderer<M, C>::drawMenu(void) {

    ImGuiIO& io = ImGui::GetIO();

    bool selected = false;

    // FPS
    this->fpsDelay += io.DeltaTime;
    if (this->fpsDelay >= 1.0f) { // update every second
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << std::setw(7)
               << ((io.Framerate > 10000.0f) ? (0.0f) : (io.Framerate));
        this->fps = stream.str();
        this->fpsDelay = 0.0f;
    }
    if (ImGui::BeginMenu("FPS")) {
        // ImGui::MenuItem("Show FPS in Window Caption", nullptr, &selected);
        // ImGui::MenuItem("Show Samples passed in Window Caption", nullptr, &selected);
        // ImGui::MenuItem("Show Primitives generated in Window Caption", nullptr, &selected);
        // ImGui::MenuItem("Copy FPS List to Clipboard", nullptr, &selected);
        if (ImGui::Button("Copy to Clipboard")) {
            ImGui::SetClipboardText(fps.data());
        }
        ImGui::EndMenu();
    }
    ImGui::Text("%s", this->fps.data());
    ImGui::Separator();

    // if (ImGui::BeginMenu("Window")) {
    //     ImGui::MenuItem("Left", nullptr, &selected);
    //     ImGui::MenuItem("Top", nullptr, &selected);
    //     ImGui::MenuItem("Width", nullptr, &selected);
    //     ImGui::MenuItem("Height", nullptr, &selected);
    //     ImGui::Separator();
    //     ImGui::MenuItem("Get Values", nullptr, &selected);
    //     ImGui::MenuItem("Set Values", nullptr, &selected);
    //     ImGui::Separator();
    //     if (ImGui::BeginMenu("Size presets")) {
    //        ImGui::MenuItem("256 x 256", nullptr, &selected);
    //        ImGui::MenuItem("512 x 512", nullptr, &selected);
    //        ImGui::MenuItem("1024 x 1024", nullptr, &selected);
    //        ImGui::MenuItem("1280 x 720", nullptr, &selected);
    //        ImGui::MenuItem("1920 x 1080", nullptr, &selected);
    //        ImGui::EndMenu();
    //    }
    //    ImGui::EndMenu();
    //}
    // ImGui::Separator();

    if (ImGui::BeginMenu("Parameters")) {
        // char filename[256];
        // if (ImGui::InputText("File Name", filename, IM_ARRAYSIZE(filename))) {
        //    // console::utility::ParamFileManager::Instance().filename = vislib::StringA(filename);
        //}
        // if (ImGui::MenuItem("Load ParamFile", nullptr, &selected)) {
        //    // console::utility::ParamFileManager::Instance().Load();
        //}
        // if (ImGui::MenuItem("Save ParamFile", nullptr, &selected)) {
        //    // console::utility::ParamFileManager::Instance().Save();
        //}
        // ImGui::Separator();
        ImGui::MenuItem("Shortcuts", nullptr, &selected);
        ImGui::EndMenu();
    }
    ImGui::Separator();

    if (ImGui::BeginMenu("Help")) {
        ImGui::MenuItem("MegaMol Help...", "h", &selected);
        ImGui::MenuItem("Report Issue...", nullptr, &selected);
        ImGui::Separator();
        ImGui::MenuItem("About...", nullptr, &selected);
        ImGui::EndMenu();
    }
    ImGui::Separator();

    bool quit = (ImGui::IsKeyDown(io.KeyMap[ImGuiKey_::ImGuiKey_Escape])) ||                    // Escape
                (ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_Q))) ||                 // 'q'
                ((io.KeyAlt) && (ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_F4)))); // Alt + F4
    if (ImGui::Button("Exit") || quit) {
        vislib::sys::Log::DefaultLog.WriteInfo(">>> GuiRenderer: Initialised shutdown of core ...");
        this->GetCoreInstance()->Shutdown();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Exit program.");
    }
}


template <class M, class C>
void GUIRenderer<M, C>::drawParameter(const core::Module& mod, core::param::ParamSlot& slot) {

    ImGuiIO& io = ImGui::GetIO();

    auto param = slot.Parameter();
    if (!param.IsNull()) {
        auto label = slot.Name().PeekBuffer();
        if (auto* p = slot.Param<core::param::BoolParam>()) {
            auto value = p->Value();
            if (ImGui::Checkbox(label, &value)) {
                p->SetValue(value);
            }
        } else if (auto* p = slot.Param<core::param::ButtonParam>()) {

            auto keyCode = p->GetKeyCode();
            auto key = this->keyCode2Key(keyCode);
            auto hotkeyPressed = (ImGui::IsKeyPressed(key)) && (keyCode.IsAltMod() == io.KeyAlt) &&
                                 (keyCode.IsCtrlMod() == io.KeyCtrl) && (keyCode.IsShiftMod() == io.KeyShift);
            // && (keyCode.IsSpecial() == io.IsSuper) - SpecialKeys(='enter','esc',...) are not SUPER key(s)

            std::string hotkeyLabel(label);
            hotkeyLabel.append(" (");
            hotkeyLabel.append(keyCode.ToStringA().PeekBuffer());
            hotkeyLabel.append(")");

            if (ImGui::Button(hotkeyLabel.data()) || hotkeyPressed) {
                p->setDirty();
            }
        } else if (auto* p = slot.Param<core::param::ColorParam>()) {
            core::param::ColorParam::Type value;
            std::memcpy(value, p->Value(), sizeof(core::param::ColorParam::Type));
            if (ImGui::ColorEdit4(label, value)) {
                p->SetValue(value);
            }
        } else if (auto* p = slot.Param<core::param::EnumParam>()) {
            // XXX: no UTF8 fanciness required here?
            auto map = p->getMap();
            auto key = p->Value();
            if (ImGui::BeginCombo(label, map[key].PeekBuffer())) {
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
        } else if (auto* p = slot.Param<core::param::FlexEnumParam>()) {
            // XXX: no UTF8 fanciness required here?
            auto value = p->Value();
            if (ImGui::BeginCombo(label, value.c_str())) {
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
        } else if (auto* p = slot.Param<core::param::FloatParam>()) {
            auto value = p->Value();
            if (ImGui::InputFloat(label, &value)) {
                p->SetValue(value);
            }
        } else if (auto* p = slot.Param<core::param::IntParam>()) {
            auto value = p->Value();
            if (ImGui::InputInt(label, &value)) {
                p->SetValue(value);
            }
        } else if (auto* p = slot.Param<core::param::Vector2fParam>()) {
            auto value = p->Value();
            if (ImGui::InputFloat2(label, value.PeekComponents())) {
                p->SetValue(value);
            }
        } else if (auto* p = slot.Param<core::param::Vector3fParam>()) {
            auto value = p->Value();
            if (ImGui::InputFloat3(label, value.PeekComponents())) {
                p->SetValue(value);
            }
        } else if (auto* p = slot.Param<core::param::Vector4fParam>()) {
            auto value = p->Value();
            if (ImGui::InputFloat4(label, value.PeekComponents())) {
                p->SetValue(value);
            }
        } else { // if (auto* p = slot.Param<core::param::StringParam>()) {
            // XXX: UTF8 conversion and allocation every frame is horrific inefficient.
            vislib::StringA valueString;
            vislib::UTF8Encoder::Encode(valueString, param->ValueString());

            size_t bufferLength = std::min(4096, (valueString.Length() + 1) * 2);
            char* buffer = new char[bufferLength];
            memcpy(buffer, valueString, valueString.Length() + 1);

            if (ImGui::InputText(slot.Name().PeekBuffer(), buffer, bufferLength)) {
                vislib::UTF8Encoder::Decode(valueString, vislib::StringA(buffer));
                param->ParseValue(valueString);
            }

            delete[] buffer;
        }
    }
}


template <class M, class C>
bool GUIRenderer<M, C>::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {

    ImGuiIO& io = ImGui::GetIO();
    auto keyIndex = static_cast<size_t>(key); // TODO: Verify mapping!
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
    io.KeySuper = mods.test(core::view::Modifier::SUPER);

    auto* cr = this->decoratedRendererSlot.template CallAs<C>();
    if (cr) {
        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;
        cr->SetInputEvent(evt);
        if ((*cr)(core::view::InputCall::FnOnKey)) return true;
    }

    return true;
}


template <class M, class C> bool GUIRenderer<M, C>::OnChar(unsigned int codePoint) {

    ImGuiIO& io = ImGui::GetIO();
    io.ClearInputCharacters();
    if (codePoint > 0 && codePoint < 0x10000) io.AddInputCharacter((unsigned short)codePoint);

    auto* cr = this->decoratedRendererSlot.template CallAs<C>();
    if (cr) {
        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::Char;
        evt.charData.codePoint = codePoint;
        cr->SetInputEvent(evt);
        if ((*cr)(core::view::InputCall::FnOnChar)) return true;
    }

    return true;
}


template <class M, class C> bool GUIRenderer<M, C>::OnMouseMove(double x, double y) {

    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2(x, y); // TODO: This is broken, since x and y are transformed by View2D class
                                // => will be fixed when screen2world transformation is available in CallRender.
    if (!ImGui::IsAnyWindowHovered()) {
        auto* cr = this->decoratedRendererSlot.template CallAs<C>();
        if (cr == NULL) return false;

        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = x;
        evt.mouseMoveData.y = y;
        cr->SetInputEvent(evt);
        if (!(*cr)(core::view::InputCall::FnOnMouseMove)) return false;
    }

    return true;
}


template <class M, class C>
bool GUIRenderer<M, C>::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {

    bool down = (action == core::view::MouseButtonAction::PRESS);
    auto buttonIndex = static_cast<size_t>(button);
    ImGuiIO& io = ImGui::GetIO();
    io.MouseDown[buttonIndex] = down;

    if (!ImGui::IsAnyWindowHovered()) {
        auto* cr = this->decoratedRendererSlot.template CallAs<C>();
        if (cr == NULL) return false;

        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;
        cr->SetInputEvent(evt);
        if (!(*cr)(core::view::InputCall::FnOnMouseButton)) return false;
    }

    return true;
}


template <class M, class C> bool GUIRenderer<M, C>::OnMouseScroll(double dx, double dy) {

    ImGuiIO& io = ImGui::GetIO();
    io.MouseWheelH += (float)dx;
    io.MouseWheel += (float)dy;

    if (!ImGui::IsAnyWindowHovered()) {
        auto* cr = this->decoratedRendererSlot.template CallAs<C>();
        if (cr == NULL) return false;

        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseScroll;
        evt.mouseScrollData.dx = dx;
        evt.mouseScrollData.dy = dy;
        cr->SetInputEvent(evt);
        if (!(*cr)(core::view::InputCall::FnOnMouseScroll)) return false;
    }

    return true;
}


template <class M, class C> int GUIRenderer<M, C>::keyCode2Key(vislib::sys::KeyCode keycode) {

    auto key = static_cast<unsigned short>(keycode.NoModKeys());

    switch (key) {
    case (97):
        return static_cast<int>(core::view::Key::KEY_A);
    case (98):
        return static_cast<int>(core::view::Key::KEY_B);
    case (99):
        return static_cast<int>(core::view::Key::KEY_C);
    case (100):
        return static_cast<int>(core::view::Key::KEY_D);
    case (101):
        return static_cast<int>(core::view::Key::KEY_E);
    case (102):
        return static_cast<int>(core::view::Key::KEY_F);
    case (103):
        return static_cast<int>(core::view::Key::KEY_G);
    case (104):
        return static_cast<int>(core::view::Key::KEY_H);
    case (105):
        return static_cast<int>(core::view::Key::KEY_I);
    case (106):
        return static_cast<int>(core::view::Key::KEY_J);
    case (107):
        return static_cast<int>(core::view::Key::KEY_K);
    case (108):
        return static_cast<int>(core::view::Key::KEY_L);
    case (109):
        return static_cast<int>(core::view::Key::KEY_M);
    case (110):
        return static_cast<int>(core::view::Key::KEY_N);
    case (111):
        return static_cast<int>(core::view::Key::KEY_O);
    case (112):
        return static_cast<int>(core::view::Key::KEY_P);
    case (113):
        return static_cast<int>(core::view::Key::KEY_Q);
    case (114):
        return static_cast<int>(core::view::Key::KEY_R);
    case (115):
        return static_cast<int>(core::view::Key::KEY_S);
    case (116):
        return static_cast<int>(core::view::Key::KEY_T);
    case (117):
        return static_cast<int>(core::view::Key::KEY_U);
    case (118):
        return static_cast<int>(core::view::Key::KEY_V);
    case (119):
        return static_cast<int>(core::view::Key::KEY_W);
    case (120):
        return static_cast<int>(core::view::Key::KEY_X);
    case (121):
        return static_cast<int>(core::view::Key::KEY_Y);
    case (122):
        return static_cast<int>(core::view::Key::KEY_Z);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_ESC)):
        return static_cast<int>(core::view::Key::KEY_ESCAPE);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_ENTER)):
        return static_cast<int>(core::view::Key::KEY_ENTER);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_TAB)):
        return static_cast<int>(core::view::Key::KEY_TAB);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_BACKSPACE)):
        return static_cast<int>(core::view::Key::KEY_BACKSPACE);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_INSERT)):
        return static_cast<int>(core::view::Key::KEY_INSERT);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_DELETE)):
        return static_cast<int>(core::view::Key::KEY_DELETE);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_RIGHT)):
        return static_cast<int>(core::view::Key::KEY_RIGHT);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_LEFT)):
        return static_cast<int>(core::view::Key::KEY_LEFT);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_DOWN)):
        return static_cast<int>(core::view::Key::KEY_DOWN);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_UP)):
        return static_cast<int>(core::view::Key::KEY_UP);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_PAGE_UP)):
        return static_cast<int>(core::view::Key::KEY_PAGE_UP);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_PAGE_DOWN)):
        return static_cast<int>(core::view::Key::KEY_PAGE_DOWN);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_HOME)):
        return static_cast<int>(core::view::Key::KEY_HOME);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_END)):
        return static_cast<int>(core::view::Key::KEY_END);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_F1)):
        return static_cast<int>(core::view::Key::KEY_F1);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_F2)):
        return static_cast<int>(core::view::Key::KEY_F2);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_F3)):
        return static_cast<int>(core::view::Key::KEY_F3);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_F4)):
        return static_cast<int>(core::view::Key::KEY_F4);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_F5)):
        return static_cast<int>(core::view::Key::KEY_F5);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_F6)):
        return static_cast<int>(core::view::Key::KEY_F6);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_F7)):
        return static_cast<int>(core::view::Key::KEY_F7);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_F8)):
        return static_cast<int>(core::view::Key::KEY_F8);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_F9)):
        return static_cast<int>(core::view::Key::KEY_F9);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_F10)):
        return static_cast<int>(core::view::Key::KEY_F10);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_F11)):
        return static_cast<int>(core::view::Key::KEY_F11);
    case (static_cast<int>(vislib::sys::KeyCode::KEY_F12)):
        return static_cast<int>(core::view::Key::KEY_F12);
    }
    return key;

    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_UNKNOWN);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_SPACE);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_APOSTROPHE);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_COMMA);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_MINUS);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_PERIOD);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_SLASH);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_0);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_1);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_2);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_3);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_4);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_5);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_6);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_7);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_8);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_9);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_SEMICOLON);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_EQUAL);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_CAPS_LOCK);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_SCROLL_LOCK);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_NUM_LOCK);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_PRINT_SCREEN);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_PAUSE);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_LEFT_BRACKET);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_BACKSLASH);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_RIGHT_BRACKET);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_GRAVE_ACCENT);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_WORLD_1);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_WORLD_2);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_F13);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_F14);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_F15);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_F16);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_F17);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_F18);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_F19);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_F20);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_F21);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_F22);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_F23);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_F24);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_F25);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_0);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_1);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_2);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_3);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_4);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_5);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_6);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_7);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_8);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_9);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_DECIMAL);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_DIVIDE);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_MULTIPLY);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_SUBTRACT);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_ADD);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_ENTER);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_KP_EQUAL);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_LEFT_SHIFT);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_LEFT_CONTROL);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_LEFT_ALT);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_LEFT_SUPER);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_RIGHT_SHIFT);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_RIGHT_CONTROL);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_RIGHT_ALT);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_RIGHT_SUPER);
    // case (static_cast<int>(vislib::sys::KeyCode::)):
    //    return static_cast<int>(core::view::Key::KEY_MENU);
}

} // end namespace gui
} // end namespace megamol

#endif // MEGAMOL_GUI_GUIRENDERER_H_INCLUDED