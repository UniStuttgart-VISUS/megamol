#include "imgui_impl_generic.h"

struct ImGui_ImplGeneric_Data {
    GenericWindow* Window;
    GenericWindow* MouseWindow;
    bool MouseJustPressed[ImGuiMouseButton_COUNT];
    GenericCursor* MouseCursors[ImGuiMouseCursor_COUNT];
    bool InstalledCallbacks;

    ImGui_ImplGeneric_Data() {
        memset(this, 0, sizeof(*this));
    }
};

static ImGui_ImplGeneric_Data* ImGui_ImplGeneric_GetBackendData() {
    return ImGui::GetCurrentContext() ? (ImGui_ImplGeneric_Data*)ImGui::GetIO().BackendPlatformUserData : NULL;
}

bool ImGui_ImplGeneric_Init(GenericWindow* window) {
    ImGuiIO& io = ImGui::GetIO();
    IM_ASSERT(io.BackendPlatformUserData == NULL && "Already initialized a platform backend!");

    // Setup backend capabilities flags
    ImGui_ImplGeneric_Data* bd = IM_NEW(ImGui_ImplGeneric_Data)();
    io.BackendPlatformUserData = (void*)bd;
    io.BackendPlatformName = "imgui_impl_generic";
    io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;
    io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;

    bd->Window = window;

    return true;
}

void ImGui_ImplGeneric_Shutdown() {
    ImGui_ImplGeneric_Data* bd = ImGui_ImplGeneric_GetBackendData();
    IM_ASSERT(bd != NULL && "No platform backend to shutdown, or already shutdown?");
    ImGuiIO& io = ImGui::GetIO();

    io.BackendPlatformName = NULL;
    io.BackendPlatformUserData = NULL;

    IM_DELETE(bd);
}

void ImGui_ImplGeneric_NewFrame(GenericWindow* window, GenericMonitor* monitor) {
    ImGuiIO& io = ImGui::GetIO();
    ImGui_ImplGeneric_Data* bd = ImGui_ImplGeneric_GetBackendData();
    IM_ASSERT(bd != NULL && "Did you call ImGui_ImplGeneric_Init()?");

    // Setup display size (every frame to accommodate for window resizing)
    int w = window->width;
    int h = window->height;
    int display_w = monitor->res_x;
    int display_h = monitor->res_y;

    io.DisplaySize = ImVec2((float)w, (float)h);
    if (w > 0 && h > 0)
        io.DisplayFramebufferScale = ImVec2((float)display_w / w, (float)display_h / h);
}
