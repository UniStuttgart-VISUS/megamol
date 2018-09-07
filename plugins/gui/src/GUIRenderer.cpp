#include "stdafx.h"
#include "GUIRenderer.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/TernaryParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"

#include "vislib/UTF8Encoder.h"

#include <imgui.h>
#include "imgui_impl_opengl3.h"

using namespace megamol;
using namespace megamol::gui;

GUIRenderer::GUIRenderer() : core::view::Renderer2DModule(), lastViewportTime(0.0) {}

GUIRenderer::~GUIRenderer() { this->Release(); }

bool GUIRenderer::create() {
    ImGui::CreateContext();
    ImGui::GetIO();

    ImGui_ImplOpenGL3_Init("#version 150");

    ImGui::StyleColorsDark();
    return true;
}

void GUIRenderer::release() {}

bool GUIRenderer::Render(core::view::CallRender2D& call) {
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

void GUIRenderer::drawMainMenu() {
    // TODO: this is still mockup stuff...
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
}

void GUIRenderer::drawParameterWindow() {
    ImGui::Begin("Parameters", &this->parameterWindowActive, ImGuiWindowFlags_AlwaysAutoResize);

    bool currentModOpen = false;
    const core::Module* currentMod = nullptr;
    ImGui::SetNextTreeNodeOpen(true);
    this->GetCoreInstance()->EnumParameters([&](const auto& mod, const auto& slot) {
        if (currentMod != &mod) {
            currentMod = &mod;
            currentModOpen = ImGui::CollapsingHeader(mod.FullName());
        }
        if (currentModOpen) {
            drawParameter(mod, slot);
        }
    });

    ImGui::End();
}

void GUIRenderer::drawParameter(const core::Module& mod, const core::param::ParamSlot& slot) {
    auto param = slot.Parameter();
    if (!param.IsNull()) {
        auto valueBuffer = this->parameterStrings[&slot];
        vislib::StringA valueString;
        vislib::UTF8Encoder::Encode(valueString, param->ValueString());
        memcpy(valueBuffer, valueString, valueString.Length() + 1);

        auto label = slot.Name().PeekBuffer();

        if (auto* p = slot.Param<core::param::BoolParam>()) {
            ImGui::Checkbox(label, reinterpret_cast<bool*>(valueBuffer));
        } else if (auto* p = slot.Param<core::param::ButtonParam>()) {
            ImGui::Button(label);
        } else if (auto* p = slot.Param<core::param::ColorParam>()) {
            ImGui::ColorEdit4(label, reinterpret_cast<float*>(valueBuffer));
        } else if (auto* p = slot.Param<core::param::EnumParam>()) {
            ImGui::BeginCombo(label, "Select...");
            ImGui::EndCombo();
        } else if (auto* p = slot.Param<core::param::FilePathParam>()) {
            // TODO: easier file handling please!
            ImGui::InputText(slot.Name().PeekBuffer(), valueBuffer, IM_ARRAYSIZE(valueBuffer));
        } else if (auto* p = slot.Param<core::param::FlexEnumParam>()) {
            ImGui::BeginCombo(label, "Select...");
            ImGui::EndCombo();
        } else if (auto* p = slot.Param<core::param::FloatParam>()) {
            ImGui::InputFloat(label, reinterpret_cast<float*>(valueBuffer));
        } else if (auto* p = slot.Param<core::param::IntParam>()) {
            ImGui::InputInt(label, reinterpret_cast<int*>(valueBuffer));
        } else if (auto* p = slot.Param<core::param::TernaryParam>()) {
            // TODO: do something smart here?
            ImGui::InputText(slot.Name().PeekBuffer(), valueBuffer, IM_ARRAYSIZE(valueBuffer));
        } else if (auto* p = slot.Param<core::param::Vector2fParam>()) {
            ImGui::InputFloat2(label, reinterpret_cast<float*>(valueBuffer));
        } else if (auto* p = slot.Param<core::param::Vector3fParam>()) {
            ImGui::InputFloat3(label, reinterpret_cast<float*>(valueBuffer));
        } else if (auto* p = slot.Param<core::param::Vector4fParam>()) {
            ImGui::InputFloat4(label, reinterpret_cast<float*>(valueBuffer));
        } else {
            ImGui::InputText(slot.Name().PeekBuffer(), valueBuffer, IM_ARRAYSIZE(valueBuffer));
        }
    }
}
