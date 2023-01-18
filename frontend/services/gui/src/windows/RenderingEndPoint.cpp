#include "RenderingEndPoint.h"

#include "mmcore/utility/log/Log.h"


megamol::gui::RenderingEndPoint::RenderingEndPoint(std::string const& window_name)
        : AbstractWindow(window_name, AbstractWindow::WINDOW_ID_RENDERING_ENDPOINT) {
    sink_.name = window_name;
    sink_.present_images = std::bind(&RenderingEndPoint::PresentImageCB, this, std::placeholders::_1);
}


//void megamol::gui::RenderingEndPoint::SetTexture(GLuint texture, uint32_t x, uint32_t y) {
//    tex_ = reinterpret_cast<ImTextureID>(static_cast<intptr_t>(texture));
//    size_ = ImVec2(x, y);
//}


void megamol::gui::RenderingEndPoint::digestChangedRequestedResources() {
    auto mouse_events_resource_ptr = frontend_resources->getOptional<frontend_resources::MouseEvents>();
    if (mouse_events_resource_ptr.has_value()) {
        frontend_resources::MouseEvents const& me = mouse_events_resource_ptr.value();
        const_cast<frontend_resources::MouseEvents&>(me).position_events.insert(
            const_cast<frontend_resources::MouseEvents&>(me).position_events.end(), position_events_.begin(),
            position_events_.end());
        const_cast<frontend_resources::MouseEvents&>(me).buttons_events.insert(
            const_cast<frontend_resources::MouseEvents&>(me).buttons_events.end(), buttons_events_.begin(),
            buttons_events_.end());
    }
    position_events_.clear();
    buttons_events_.clear();
}


bool megamol::gui::RenderingEndPoint::Draw() {
    static const char* current_item = nullptr;
    megamol::frontend::ImagePresentation_Service::EntryPointRenderFunctions entry_point;
    /*if (ImGui::BeginMainMenuBar()) {


        ImGui::EndMainMenuBar();
    }*/

    auto& img_pres_ep_resource_ptr = frontend_resources->get<frontend_resources::ImagePresentationEntryPoints>();
    auto mouse_events_resource_ptr = frontend_resources->getOptional<frontend_resources::MouseEvents>();

    bool isSelected = false;
    if (ImGui::BeginCombo("Views", current_item)) {
        for (auto const& item : entry_points_) {
            if (ImGui::Selectable(item.first.c_str(), &isSelected)) {
                if (current_item != nullptr) {
                    img_pres_ep_resource_ptr.unbind_sink_entry_point(this->Name(), item.first);
                }
                current_item = item.first.c_str();
                entry_point = item.second;
                img_pres_ep_resource_ptr.bind_sink_entry_point(this->Name(), item.first);
            }
        }
        ImGui::EndCombo();
    }

    /*ImGui::Text("RenderEndPoint");
    ImGui::Spacing();*/

    //entry_point
    if (current_item != nullptr) {
        if (mouse_events_resource_ptr.has_value() && ImGui::IsWindowHovered()) {
            ImVec2 mousePositionAbsolute = ImGui::GetMousePos();
            ImVec2 screenPositionAbsolute = ImGui::GetItemRectMin();
            ImVec2 mousePositionRelative = ImVec2(
                mousePositionAbsolute.x - screenPositionAbsolute.x, mousePositionAbsolute.y - screenPositionAbsolute.y);
            /*frontend_resources::MouseEvents const& me = mouse_events_resource_ptr.value();
            const_cast<frontend_resources::MouseEvents&>(me).position_events.emplace_back(
                std::make_tuple(mousePositionRelative.x, mousePositionRelative.y));*/
            position_events_.emplace_back(std::make_tuple(mousePositionRelative.x, mousePositionRelative.y));

            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                if (ImGui::IsKeyDown(ImGuiKey_ModAlt)) {
                    frontend_resources::MouseButton btn = frontend_resources::MouseButton::BUTTON_LEFT;
                    frontend_resources::MouseButtonAction btnaction = frontend_resources::MouseButtonAction::PRESS;


                    frontend_resources::Modifiers btnmods;
                    btnmods |= frontend_resources::Modifier::ALT;

                    /*const_cast<frontend_resources::MouseEvents&>(me).buttons_events.emplace_back(
                        std::make_tuple(btn, btnaction, btnmods));*/
                    buttons_events_.emplace_back(std::make_tuple(btn, btnaction, btnmods));
                }
            }

            /*core::utility::log::Log::DefaultLog.WriteInfo(
                "Window coord %f %f", mousePositionRelative.x, mousePositionRelative.y);*/
        }
        for (auto& image : images_) {
            ImGui::Image(image.referenced_image_handle, ImVec2{(float)image.size.width, (float)image.size.height},
                ImVec2(0, 1), ImVec2(1, 0));
        }

        //auto ep = img_pres_ep_resource_ptr.get_entry_point(current_item);
        //if (ep.has_value()) {
        //    frontend_resources::EntryPoint& ep_v = ep.value();

        //    /*ep_v.entry_point_data->update();

        //    ep_v.execute(ep_v.modulePtr, ep_v.entry_point_resources, ep_v.execution_result_image);*/

        //    ImGui::Image(ep_v.execution_result_image.referenced_image_handle,
        //        ImVec2{(float)ep_v.execution_result_image.size.width, (float)ep_v.execution_result_image.size.height},
        //        ImVec2(0, 1), ImVec2(1, 0));
        //    //ImGui::Image(tex_, size_, ImVec2(0, 1), ImVec2(1, 0));
        //}
    }
    /*if (ImGui::Begin("RenderingEndPoint")) {
        ImGui::Text("RenderEndPoint");
        ImGui::Spacing();

        ImGui::Image(tex_, size_, ImVec2(0, 1), ImVec2(1, 0));
    }
    ImGui::End();*/

    return true;
}
