#include "stdafx.h"
#include "GUIRenderer.h"

#include "mmcore/CoreInstance.h"

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
    drawParameterWindow();

    // Render frame.
    glViewport(0, 0, viewportWidth, viewportHeight);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    lastViewportTime = viewportTime;

    return true;
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

        ImGui::InputText(slot.Name().PeekBuffer(), valueBuffer, IM_ARRAYSIZE(valueBuffer));
    }
}
