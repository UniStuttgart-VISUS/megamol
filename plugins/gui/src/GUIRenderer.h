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

    /** The decorated renderer caller slot */
    core::CallerSlot decoratedRendererSlot;

    double lastViewportTime;

    bool parameterWindowOpen;
};

typedef GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D> GUIRenderer2D;
typedef GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D> GUIRenderer3D;

template <>
inline GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::GUIRenderer()
    : decoratedRendererSlot("decoratedRenderer", "Connects to another 2D Renderer being decorated")
    , lastViewportTime(0.0) {

    this->decoratedRendererSlot.SetCompatibleCall<core::view::CallRender2DDescription>();
    this->MakeSlotAvailable(&this->decoratedRendererSlot);
}

template <> inline const char* GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::ClassName(void) {
    return "GUIRenderer2D";
}

template <> inline const char* GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::ClassName(void) {
    return "GUIRenderer3D";
}

template <>
inline GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::GUIRenderer()
    : decoratedRendererSlot("decoratedRenderer", "Connects to another 3D Renderer being decorated")
    , lastViewportTime(0.0) {

    this->decoratedRendererSlot.SetCompatibleCall<core::view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->decoratedRendererSlot);
}

template <class M, class C> GUIRenderer<M, C>::~GUIRenderer() { this->Release(); }

template <class M, class C> bool GUIRenderer<M, C>::create() {
    ImGui::CreateContext();
    ImGui::GetIO();

    ImGui_ImplOpenGL3_Init("#version 150");

    ImGui::StyleColorsDark();
    return true;
}
template <class M, class C> void GUIRenderer<M, C>::release() {}

template <class M, class C>
bool GUIRenderer<M, C>::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {
    ImGuiIO& io = ImGui::GetIO();
    auto keyIndex = static_cast<size_t>(key); // TODO: verify mapping!
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
    // io.KeySuper =  mods.test(core::view::Modifier::SUPER)
    return true;
}

template <class M, class C> bool GUIRenderer<M, C>::OnChar(unsigned int codePoint) {
    ImGuiIO& io = ImGui::GetIO();
    if (codePoint > 0 && codePoint < 0x10000) io.AddInputCharacter((unsigned short)codePoint);
    return true;
}

template <class M, class C>
bool GUIRenderer<M, C>::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    bool down = (action == core::view::MouseButtonAction::PRESS);
    auto buttonIndex = static_cast<size_t>(button);
    ImGuiIO& io = ImGui::GetIO();
    io.MouseDown[buttonIndex] = down;
    return true;
}

template <class M, class C> bool GUIRenderer<M, C>::OnMouseMove(double x, double y) {
    ImGuiIO& io = ImGui::GetIO();
    io.MousePos = ImVec2(x, y); // TODO: this is broken, since x and y are transformed by View classes.
    return true;
}

template <class M, class C> bool GUIRenderer<M, C>::OnMouseScroll(double dx, double dy) {
    ImGuiIO& io = ImGui::GetIO();
    io.MouseWheelH += (float)dx;
    io.MouseWheel += (float)dy;
    return true;
}

template <class M, class C> void GUIRenderer<M, C>::drawMainMenu() {
#if 0 // TODO: this is still mockup stuff...
    bool a, b, c;
    bool d, e, f;
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            ImGui::MenuItem("New", nullptr, &a);
            ImGui::MenuItem("Open", nullptr, &a);
            ImGui::EndMenu();
            ImGui::MenuItem("Save", nullptr, &a);
            ImGui::MenuItem("Save as...", nullptr, &a);
            ImGui::EndMenu();
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
#endif
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
        // TODO: set dummy bounding box?
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
    auto viewportTime = call.InstanceTime();

    // Start the frame
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(viewportWidth, viewportHeight);
    io.DisplayFramebufferScale = ImVec2(1.0, 1.0);
    io.DeltaTime = viewportTime - lastViewportTime;
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

    lastViewportTime = viewportTime;

    return true;
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

    ImGui::End();
}

template <class M, class C>
void GUIRenderer<M, C>::drawParameter(const core::Module& mod, core::param::ParamSlot& slot) {
    auto param = slot.Parameter();
    if (!param.IsNull()) {
        auto label = slot.Name().PeekBuffer();
        if (auto* p = slot.Param<core::param::BoolParam>()) {
            auto value = p->Value();
            if (ImGui::Checkbox(label, &value)) {
                p->SetValue(value);
            }
        } else if (auto* p = slot.Param<core::param::ButtonParam>()) {
            // TODO: fiddle with key code (no getter and it is private - wtf?)
            if (ImGui::Button(label)) {
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


} // end namespace gui
} // end namespace megamol

#endif // MEGAMOL_GUI_GUIRENDERER_H_INCLUDED