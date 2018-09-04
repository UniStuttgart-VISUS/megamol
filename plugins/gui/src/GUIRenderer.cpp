#include "stdafx.h"
#include "GUIRenderer.h"

#include <imgui.h>
#include "imgui_impl_opengl3.h"

using namespace megamol;
using namespace megamol::gui;

void do_stuff() {
    ImGui::Text("Hello, world %d", 123);
    if (ImGui::Button("Save")) {
        // do stuff
    }

    float f;
    ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
}

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
    do_stuff();

    // Render frame.
    glViewport(0, 0, viewportWidth, viewportHeight);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    lastViewportTime = viewportTime;

    return true;
}
