#include "stdafx.h"
#include "GUIRenderer.h"

#include <imgui.h>

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

GUIRenderer::GUIRenderer() : core::view::Renderer2DModule() {}

GUIRenderer::~GUIRenderer() { this->Release(); }

bool GUIRenderer::create() {
    ImGui::CreateContext();
    ImGui::GetIO();

    // ImGui_ImplGlfw_InitForOpenGL(window, true);
    // ImGui_ImplOpenGL3_Init(glsl_version);

    ImGui::StyleColorsDark();
    return true;
}

void GUIRenderer::release() {}

bool GUIRenderer::Render(core::view::CallRender2D& call) {
    // Start the frame
    // ImGui_ImplOpenGL3_NewFrame();
    // ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Construct frame, i.e., geometry and stuff.
    do_stuff();

    // Render frame.
    ImGui::Render();
    // glViewport(0, 0, display_w, display_h);
    // ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    return true;
}
