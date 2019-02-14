#ifndef MEGAMOL_GUI_GUIRENDERER_H_INCLUDED
#define MEGAMOL_GUI_GUIRENDERER_H_INCLUDED

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
    void drawMainMenu();

    /**
     * Draws a parameter window.
     */
    void drawParameterWindow();

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

    bool parameterWindowOpen;
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

    const char* glsl_version = "#version 150";

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
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

    ImGui::StyleColorsDark();
    ImGui_ImplOpenGL3_Init(glsl_version);

    return true;
}


template <class M, class C> void GUIRenderer<M, C>::release() {

    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();
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

    // Ignore 'ESC' and 'q'
    if ((ImGui::IsKeyDown(io.KeyMap[ImGuiKey_::ImGuiKey_Escape])) ||
        (ImGui::IsKeyDown(static_cast<int>(core::view::Key::KEY_Q)))) {
        return false;
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

    // Start the frame
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(viewportWidth, viewportHeight);
    io.DisplayFramebufferScale = ImVec2(1.0, 1.0);
    io.DeltaTime = static_cast<float>(call.LastFrameTime() / 1000.0); // in milliseconds

    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();

    // Construct frame, i.e., geometry and stuff.
    // XXX: drawOtherStuff();
    drawMainMenu();
    drawParameterWindow();

    // Render frame.
    glViewport(0, 0, viewportWidth, viewportHeight);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return true;
}


template <class M, class C> void GUIRenderer<M, C>::drawMainMenu() {

    bool a, b, c;
    bool d, e, f, g;
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            ImGui::MenuItem("New", "x", false);
            ImGui::MenuItem("Open", nullptr, &a);
            ImGui::MenuItem("Save", nullptr, &a);
            ImGui::MenuItem("Save as...", nullptr, &a);
            ImGui::MenuItem("Exit", nullptr, &a);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit")) {
            ImGui::MenuItem("Cut", nullptr, &a);
            ImGui::MenuItem("Copy", nullptr, &a);
            ImGui::MenuItem("Paste", nullptr, &a);
            ImGui::MenuItem("Delete", nullptr, &a);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Parameter inspector", nullptr, &a);
            ImGui::MenuItem("Node editor", nullptr, &b);
            ImGui::MenuItem("Console", nullptr, &c);
            ImGui::Separator();
            ImGui::MenuItem("Settings...", nullptr, &c);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Help")) {
            ImGui::MenuItem("MegaMol Help...", nullptr, &e);
            ImGui::MenuItem("Report Issue...", nullptr, &e);
            ImGui::Separator();
            ImGui::MenuItem("About...", nullptr, &f);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}


template <class M, class C> void GUIRenderer<M, C>::drawParameterWindow() {

    ImGui::Begin("Parameters", &this->parameterWindowOpen, ImGuiWindowFlags_AlwaysAutoResize);

    const core::Module* currentMod = nullptr;
    bool currentModOpen = false;
    this->GetCoreInstance()->EnumParameters([&, this](const auto& mod, auto& slot) {
        if (currentMod != &mod) {
            currentMod = &mod;
            // Set to "open" by default.
            auto headerId = ImGui::GetID(mod.FullName());
            int headerState = ImGui::GetStateStorage()->GetInt(headerId, 1);
            ImGui::GetStateStorage()->SetInt(headerId, headerState);
            currentModOpen = ImGui::CollapsingHeader(mod.FullName());
        }
        if (currentModOpen) {
            this->drawParameter(mod, slot);
        }
    });

    // TEMP ///////////////////////////////////////////////////////////////////////////////////////////
    ImGuiIO& io = ImGui::GetIO();
    if (ImGui::TreeNode("Keyboard, Mouse & Navigation State")) {
        if (ImGui::IsMousePosValid())
            ImGui::Text("Mouse pos: (%g, %g)", io.MousePos.x, io.MousePos.y);
        else
            ImGui::Text("Mouse pos: <INVALID>");
        ImGui::Text("Mouse delta: (%g, %g)", io.MouseDelta.x, io.MouseDelta.y);
        ImGui::Text("Mouse down:");
        for (int i = 0; i < IM_ARRAYSIZE(io.MouseDown); i++)
            if (io.MouseDownDuration[i] >= 0.0f) {
                ImGui::SameLine();
                ImGui::Text("b%d (%.02f secs)", i, io.MouseDownDuration[i]);
            }
        ImGui::Text("Mouse clicked:");
        for (int i = 0; i < IM_ARRAYSIZE(io.MouseDown); i++)
            if (ImGui::IsMouseClicked(i)) {
                ImGui::SameLine();
                ImGui::Text("b%d", i);
            }
        ImGui::Text("Mouse dbl-clicked:");
        for (int i = 0; i < IM_ARRAYSIZE(io.MouseDown); i++)
            if (ImGui::IsMouseDoubleClicked(i)) {
                ImGui::SameLine();
                ImGui::Text("b%d", i);
            }
        ImGui::Text("Mouse released:");
        for (int i = 0; i < IM_ARRAYSIZE(io.MouseDown); i++)
            if (ImGui::IsMouseReleased(i)) {
                ImGui::SameLine();
                ImGui::Text("b%d", i);
            }
        ImGui::Text("Mouse wheel: %.1f", io.MouseWheel);

        ImGui::Text("Keys down:");
        for (int i = 0; i < IM_ARRAYSIZE(io.KeysDown); i++)
            if (io.KeysDownDuration[i] >= 0.0f) {
                ImGui::SameLine();
                ImGui::Text("%d (%.02f secs)", i, io.KeysDownDuration[i]);
            }
        ImGui::Text("Keys pressed:");
        for (int i = 0; i < IM_ARRAYSIZE(io.KeysDown); i++)
            if (ImGui::IsKeyPressed(i)) {
                ImGui::SameLine();
                ImGui::Text("%d", i);
            }
        ImGui::Text("Keys release:");
        for (int i = 0; i < IM_ARRAYSIZE(io.KeysDown); i++)
            if (ImGui::IsKeyReleased(i)) {
                ImGui::SameLine();
                ImGui::Text("%d", i);
            }
        ImGui::Text("Keys mods: %s%s%s%s", io.KeyCtrl ? "CTRL " : "", io.KeyShift ? "SHIFT " : "",
            io.KeyAlt ? "ALT " : "", io.KeySuper ? "SUPER " : "");

        ImGui::Text("NavInputs down:");
        for (int i = 0; i < IM_ARRAYSIZE(io.NavInputs); i++)
            if (io.NavInputs[i] > 0.0f) {
                ImGui::SameLine();
                ImGui::Text("[%d] %.2f", i, io.NavInputs[i]);
            }
        ImGui::Text("NavInputs pressed:");
        for (int i = 0; i < IM_ARRAYSIZE(io.NavInputs); i++)
            if (io.NavInputsDownDuration[i] == 0.0f) {
                ImGui::SameLine();
                ImGui::Text("[%d]", i);
            }
        ImGui::Text("NavInputs duration:");
        for (int i = 0; i < IM_ARRAYSIZE(io.NavInputs); i++)
            if (io.NavInputsDownDuration[i] >= 0.0f) {
                ImGui::SameLine();
                ImGui::Text("[%d] %.2f", i, io.NavInputsDownDuration[i]);
            }

        ImGui::Button("Hovering me sets the\nkeyboard capture flag");
        if (ImGui::IsItemHovered()) ImGui::CaptureKeyboardFromApp(true);
        ImGui::SameLine();
        ImGui::Button("Holding me clears the\nthe keyboard capture flag");
        if (ImGui::IsItemActive()) ImGui::CaptureKeyboardFromApp(false);

        ImGui::TreePop();
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    ImGui::End();
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
        } else {
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