#pragma once

#include <map>
#include <stdint.h>

#include <glad/gl.h>

#include "AbstractWindow.h"

#include "ImagePresentationEntryPoints.h"
#include "ImagePresentation_Service.hpp"

#include "KeyboardMouse_Events.h"

namespace megamol::gui {
class RenderingEndPoint : public AbstractWindow {
public:
    std::vector<std::string> requested_lifetime_resources() const override {
        auto res = AbstractWindow::requested_lifetime_resources();
        res.push_back("ImagePresentationEntryPoints");
        res.push_back("optional<MouseEvents>");
        return res;
    }

    void setRequestedResources(std::shared_ptr<frontend_resources::FrontendResourcesMap> const& resources) override {
        AbstractWindow::setRequestedResources(resources);
        auto& img_pres_ep_resource_ptr = frontend_resources->get<frontend_resources::ImagePresentationEntryPoints>();

        auto sub_func = [&](frontend_resources::ImagePresentationEntryPoints::SubscriptionEvent const& event,
                            std::vector<std::any> const& args) -> void {
            switch (event) {
            case frontend_resources::ImagePresentationEntryPoints::SubscriptionEvent::Add: {
                entry_points_.insert(std::make_pair(std::any_cast<std::string>(args[0]),
                    std::any_cast<frontend::ImagePresentation_Service::EntryPointRenderFunctions>(args[1])));
            } break;
            case frontend_resources::ImagePresentationEntryPoints::SubscriptionEvent::Remove: {
                entry_points_.erase(std::any_cast<std::string>(args[0]));
            } break;
            case frontend_resources::ImagePresentationEntryPoints::SubscriptionEvent::Rename: {
                auto func = entry_points_[std::any_cast<std::string>(args[0])];
                entry_points_.erase(std::any_cast<std::string>(args[0]));
                entry_points_.insert(std::make_pair(std::any_cast<std::string>(args[1]), func));
            } break;
            case frontend_resources::ImagePresentationEntryPoints::SubscriptionEvent::Clear:
            default:
                break;
            }
        };

        img_pres_ep_resource_ptr.subscribe_to_entry_point_changes(sub_func);
        img_pres_ep_resource_ptr.add_sink(sink_);
    }

    void digestChangedRequestedResources() override;

    explicit RenderingEndPoint(const std::string& window_name);

    //virtual ~RenderingEndPoint() {
    //    auto& img_pres_ep_resource_ptr = frontend_resources->get<frontend_resources::ImagePresentationEntryPoints>();
    //    img_pres_ep_resource_ptr.remove_sink(sink_.name);
    //}

    //void SetTexture(GLuint texture, uint32_t x, uint32_t y);

    bool Draw() override;

    void PresentImageCB(std::vector<frontend_resources::ImageWrapper> const& images) {
        images_ = images;
    }

private:
    /*ImTextureID tex_;
    ImVec2 size_;*/
    std::map<std::string, frontend::ImagePresentation_Service::EntryPointRenderFunctions> entry_points_;
    std::vector<frontend_resources::ImageWrapper> images_;
    frontend_resources::ImagePresentationSink sink_;

    std::vector<std::tuple<frontend_resources::MouseButton, frontend_resources::MouseButtonAction,
        frontend_resources::Modifiers>>
        buttons_events_;
    std::vector<std::tuple<double, double>> position_events_;
};
} // namespace megamol::gui
