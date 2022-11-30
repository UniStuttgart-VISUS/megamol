#include "RenderingEndPoint.h"


megamol::gui::RenderingEndPoint::RenderingEndPoint(std::string const& window_name)
        : AbstractWindow(window_name, AbstractWindow::WINDOW_IF_RENDERING_ENDPOINT) {}


void megamol::gui::RenderingEndPoint::SetTexture(GLuint texture, uint32_t x, uint32_t y) {
    tex_ = reinterpret_cast<ImTextureID>(static_cast<intptr_t>(texture));
    size_ = ImVec2(x, y);
}


bool megamol::gui::RenderingEndPoint::Draw() {
    static const char* current_item = nullptr;
    megamol::frontend::ImagePresentation_Service::EntryPointRenderFunctions entry_point;
    /*if (ImGui::BeginMainMenuBar()) {


        ImGui::EndMainMenuBar();
    }*/

    auto& img_pres_ep_resource_ptr = resources_[0].getResource<frontend_resources::ImagePresentationEntryPoints>();

    bool isSelected = false;
    if (ImGui::BeginCombo("Views", current_item)) {
        for (auto const& item : entry_points_) {
            if (ImGui::Selectable(item.first.c_str(), &isSelected)) {
                current_item = item.first.c_str();
                entry_point = item.second;
            }
        }
        ImGui::EndCombo();
    }

    /*ImGui::Text("RenderEndPoint");
    ImGui::Spacing();*/

    //entry_point
    if (current_item != nullptr) {
        auto ep = img_pres_ep_resource_ptr.get_entry_point(current_item);
        if (ep.has_value()) {
            frontend_resources::EntryPoint& ep_v = ep.value();

            ep_v.entry_point_data->update();

            ep_v.execute(ep_v.modulePtr, ep_v.entry_point_resources, ep_v.execution_result_image);

            ImGui::Image(ep_v.execution_result_image.referenced_image_handle,
                ImVec2{(float)ep_v.execution_result_image.size.width, (float)ep_v.execution_result_image.size.height},
                ImVec2(0, 1), ImVec2(1, 0));
            //ImGui::Image(tex_, size_, ImVec2(0, 1), ImVec2(1, 0));
        }
    }
    /*if (ImGui::Begin("RenderingEndPoint")) {
        ImGui::Text("RenderEndPoint");
        ImGui::Spacing();

        ImGui::Image(tex_, size_, ImVec2(0, 1), ImVec2(1, 0));
    }
    ImGui::End();*/

    return true;
}
