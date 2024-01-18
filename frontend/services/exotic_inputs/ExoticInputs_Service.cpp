/*
 * ExoticInputs_Service.cpp
 *
 * Copyright (C) 2022 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "ExoticInputs_Service.hpp"

#include "GamepadState.h"
#include "ModuleGraphSubscription.h"

#include "GUIRegisterWindow.h" // register UI window for remote control
#include "camera_controllers.h"
#include "imgui_stdlib.h"
#include "mmcore/Module.h"
#include "mmcore/view/AbstractViewInterface.h"

// local logging wrapper for your convenience until central MegaMol logger established
#include "mmcore/utility/log/Log.h"

#include <algorithm>
#include <cmath>

static const std::string service_name = "ExoticInputs_Service: ";
static void log(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}

static void log_error(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteError(msg.c_str());
}

static void log_warning(std::string const& text) {
    const std::string msg = service_name + text;
    megamol::core::utility::log::Log::DefaultLog.WriteWarn(msg.c_str());
}


namespace megamol {
namespace frontend {

ExoticInputs_Service::ExoticInputs_Service() {
    // init members to default states
}

ExoticInputs_Service::~ExoticInputs_Service() {
    // clean up raw pointers you allocated with new, which is bad practice and nobody does
}

bool ExoticInputs_Service::init(void* configPtr) {
    if (configPtr == nullptr)
        return false;

    return init(*static_cast<Config*>(configPtr));
}

bool ExoticInputs_Service::init(const Config& config) {
    // initialize your service and its provided resources using config parameters
    // for now, you dont need to worry about your service beeing initialized or closed multiple times
    // init() and close() only get called once in the lifetime of each service object
    // but maybe more instances of your service will get created? this may be relevant for central resources you manage (like libraries, network connections).

    m_providedResourceReferences = {};

    m_requestedResourcesNames = {
        "optional<Connected_Gamepads>",
        "optional<GUIRegisterWindow>",
        frontend_resources::MegaMolGraph_SubscriptionRegistry_Req_Name,
    };

    log("initialized successfully");
    return true;
}

void ExoticInputs_Service::close() {}

std::vector<FrontendResource>& ExoticInputs_Service::getProvidedResources() {
    return m_providedResourceReferences;
}

const std::vector<std::string> ExoticInputs_Service::getRequestedResourceNames() const {
    return m_requestedResourcesNames;
}

void ExoticInputs_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    m_requestedResourceReferences = resources;

    frontend_resources::ModuleGraphSubscription subscription("Exotic_Inputs");

    //subscription.EnableEntryPoint = [&](core::ModuleInstance_t const& module_inst) {
    //};
    //subscription.DisableEntryPoint = [&](core::ModuleInstance_t const& module_inst) {
    //};

    subscription.AddModule = [&](core::ModuleInstance_t const& module_inst) {
        auto& view_module_name = module_inst.request.id;
        // this cast will never fail but the modulePtr should be valid nonetheless
        // no need to check right entry point because they represent the same view module
        auto* ptr = static_cast<megamol::core::Module*>(module_inst.modulePtr.get());
        if (!ptr) {
            log_error("entry point " + view_module_name +
                      " does not seem to have a valid megamol::core::Module* (is nullptr).");
            return true;
        }

        // if the entry point is not a 3d view there is no point in doing stereo for it
        const auto* view = dynamic_cast<megamol::core::view::AbstractViewInterface*>(ptr);
        if (view == nullptr || view->GetViewDimension() != core::view::AbstractViewInterface::ViewDimension::VIEW_3D) {
            log_error("entry point " + view_module_name +
                      " does not seem to be a supported 3D View Type. Not using it to inject camera manipulation via "
                      "exotic inputs.");
            return true;
        }

        m_view3d_modules.emplace(view_module_name, ptr);

        if (!m_controlled_view3d.has_value())
            m_controlled_view3d = m_view3d_modules.begin()->first;

        return true;
    };

    subscription.DeleteModule = [&](core::ModuleInstance_t const& module_inst) {
        auto id = module_inst.request.id;

        if (m_view3d_modules.count(module_inst.request.id)) {
            m_view3d_modules.erase(id);

            if (m_controlled_view3d == id) {
                m_controlled_view3d =
                    m_view3d_modules.empty() ? std::nullopt : std::make_optional(m_view3d_modules.begin()->first);
            }
        }

        return true;
    };

    subscription.RenameModule = [&](std::string const& old_name, std::string const& new_name,
                                    core::ModuleInstance_t const& module_inst) {
        auto old_id = old_name;
        auto new_id = new_name;

        if (old_id != new_id && m_view3d_modules.count(old_id)) {

            auto v = m_view3d_modules.at(old_id);
            m_view3d_modules.erase(old_id);
            m_view3d_modules.emplace(new_id, v);

            if (m_controlled_view3d == old_id) {
                m_controlled_view3d = new_id;
            }
        }

        return true;
    };

    //subscription.AddParameters =
    //    [&](std::vector<megamol::frontend_resources::ModuleGraphSubscription::ParamSlotPtr> const& param_slots) {
    //    };
    //subscription.RemoveParameters =
    //    [&](std::vector<megamol::frontend_resources::ModuleGraphSubscription::ParamSlotPtr> const& param_slots) {
    //    };
    //subscription.ParameterChanged =
    //    [&](megamol::frontend_resources::ModuleGraphSubscription::ParamSlotPtr const& param_slot,
    //        std::string const& new_value) {
    //    };
    //subscription.ParameterPresentationChanged =
    //    [&](megamol::frontend_resources::ModuleGraphSubscription::ParamSlotPtr const& param_slot) {
    //    };
    //subscription.AddCall = [&](core::CallInstance_t const& call_inst) {
    //};
    //subscription.DeleteCall = [&](core::CallInstance_t const& call_inst) {
    //};

    auto& megamolgraph_subscription = const_cast<frontend_resources::MegaMolGraph_SubscriptionRegistry&>(
        m_requestedResourceReferences[2].getResource<frontend_resources::MegaMolGraph_SubscriptionRegistry>());
    megamolgraph_subscription.subscribe(subscription);

    gamepad_window();
}

void ExoticInputs_Service::updateProvidedResources() {}

void ExoticInputs_Service::digestChangedRequestedResources() {
    auto maybe_gamepad_resource =
        m_requestedResourceReferences[0].getOptionalResource<megamol::frontend_resources::Connected_Gamepads>();

    auto threshold = [&](const float f) -> float { return (std::fabs(f) < m_axis_threshold) ? (0.0f) : (f); };
    auto thresholdv2 = [&](const glm::vec2 v) -> glm::vec2 { return glm::vec2(threshold(v.x), threshold(v.y)); };
    auto thresholdv3 = [&](const glm::vec3 v) -> glm::vec3 {
        return glm::vec3(threshold(v.x), threshold(v.y), threshold(v.z));
    };
    auto norm = [](const float f) -> float { return (f + 1.0f) * 0.5f; };

    auto apply_pose_controls = [&](const camera_controllers::Pose pose, const PoseManipulator mode,
                                   const megamol::frontend_resources::GamepadState pad, const float scale,
                                   const glm::vec3 center) -> camera_controllers::Pose {
        using Pad = megamol::frontend_resources::GamepadState;

        const glm::vec2 stick_left = thresholdv2(glm::vec2{pad.axis(Pad::Axis::LEFT_X), pad.axis(Pad::Axis::LEFT_Y)});
        const glm::vec2 stick_right =
            thresholdv2(glm::vec2{pad.axis(Pad::Axis::RIGHT_X), pad.axis(Pad::Axis::RIGHT_Y)});
        const float vertial = threshold(
            norm(pad.axis(Pad::Axis::LEFT_TRIGGER)) * (-1.0f) + norm(pad.axis(Pad::Axis::RIGHT_TRIGGER)) * (1.0f));

        const float rad_per_axis_unit = 0.1f;
        const float trans_per_axis_unit = 0.05f * scale;

        const glm::vec2 rotation = rad_per_axis_unit * (stick_right * glm::vec2{-1.0f, 1.0f});

        const glm::vec3 translation = trans_per_axis_unit * glm::vec3{stick_left.x, vertial, -stick_left.y};

        const glm::vec3 rotation_center = center;
        const float orbit_distance = stick_left.y * trans_per_axis_unit;

        using namespace camera_controllers;
        switch (mode) {
        case PoseManipulator::Arcball:
            return arcball{rotation_center}.apply(
                orbit_altitude{rotation_center, orbit_altitude::Mode::Relative_Factor}.apply(pose, orbit_distance),
                rotation);
            break;
        //case PoseManipulator::Turntable:
        //    return turntable{}.apply(
        //        orbit_altitude{rotation_center, orbit_altitude::Mode::Relative_Factor}.apply(pose, orbit_distance),
        //        rotation);
        //break;
        case PoseManipulator::FPS:
            return fps{}.apply(pose, translation, glm::vec2{-rotation.y, rotation.x} * 0.1f);
            break;
        default:
            break;
        }
        return pose;
    };

    if (!m_controlled_view3d.has_value()) {
        return;
    }

    auto view3d_ptr = m_view3d_modules.at(m_controlled_view3d.value());

    auto* view = const_cast<megamol::core::view::AbstractViewInterface*>(
        dynamic_cast<megamol::core::view::AbstractViewInterface*>(static_cast<megamol::core::Module*>(view3d_ptr)));

    if (!view) {
        log_error("view reference in exotic inputs service did not resolve to View3D");
        return;
    }

    auto bbox = view->GetBoundingBoxes();
    auto center = glm::vec3{bbox.BoundingBox().CalcCenter().GetX(), bbox.BoundingBox().CalcCenter().GetY(),
        bbox.BoundingBox().CalcCenter().GetZ()};
    auto scale = view->GetBoundingBoxes().BoundingBox().LongestEdge();

    if (!maybe_gamepad_resource.has_value()) {
        return;
    }

    auto& connected_gamepads = maybe_gamepad_resource.value().get();

    // to avoid switching camera mode every frame when button pressed for several frames, use this counter
    static size_t mode_changed = 0;
    mode_changed++;
    const size_t button_threshold = 60 * 1; // allow switching mode roughly every second

    for (const auto& pad_ : connected_gamepads.gamepads) {
        auto& pad = pad_.get();

        // switch active entry point and camera control
        if ((pad.pressed(megamol::frontend_resources::GamepadState::Button::DPAD_LEFT) ||
                pad.pressed(megamol::frontend_resources::GamepadState::Button::DPAD_RIGHT)) &&
            mode_changed > button_threshold) {
            mode_changed = 0;

            if (m_controlled_view3d.has_value() && m_view3d_modules.size() > 1) {
                auto it = m_view3d_modules.find(m_controlled_view3d.value());

                if (pad.pressed(megamol::frontend_resources::GamepadState::Button::DPAD_LEFT)) {
                    if (it == m_view3d_modules.begin()) {
                        it = (--m_view3d_modules.end());
                    } else {
                        it--;
                    }
                }

                if (pad.pressed(megamol::frontend_resources::GamepadState::Button::DPAD_RIGHT)) {
                    if ((++it) == m_view3d_modules.end()) {
                        it = m_view3d_modules.begin();
                    }
                }

                m_controlled_view3d = it->first;
            }
        }

        // reset camera via view
        if (pad.pressed(megamol::frontend_resources::GamepadState::Button::LEFT_BUMPER) &&
            mode_changed > button_threshold) {
            mode_changed = 0;

            auto move_by = scale * 3;

            glm::vec3 position = {center.x - move_by, center.y, center.z};
            glm::vec3 forward = glm::normalize(center - position);
            glm::vec3 up = {0.0f, 1.0f, 0.0f};
            megamol::core::view::Camera::Pose new_pose(position, forward, up, glm::cross(forward, up));

            auto cam = view->GetCamera();
            cam.setPose(new_pose);
            view->SetCamera(cam);
        }

        // switch camera control mode
        if (pad.pressed(megamol::frontend_resources::GamepadState::Button::RIGHT_BUMPER) &&
            mode_changed > button_threshold) {
            mode_changed = 0;

            m_manipulation_mode = static_cast<PoseManipulator>((static_cast<unsigned int>(m_manipulation_mode) + 1) %
                                                               static_cast<unsigned int>(PoseManipulator::COUNT));
        }

        auto camera = view->GetCamera();
        auto in_pose = camera.getPose();
        auto out_pose = apply_pose_controls(in_pose, m_manipulation_mode, pad, scale, center);
        camera.setPose(out_pose);
        view->SetCamera(camera);
    }
}

void ExoticInputs_Service::resetProvidedResources() {}

void ExoticInputs_Service::preGraphRender() {}

void ExoticInputs_Service::postGraphRender() {}

void ExoticInputs_Service::gamepad_window() const {
    auto maybe_gamepad_resource =
        m_requestedResourceReferences[0].getOptionalResource<megamol::frontend_resources::Connected_Gamepads>();
    auto maybe_window_resource =
        m_requestedResourceReferences[1].getOptionalResource<megamol::frontend_resources::GUIRegisterWindow>();

    if (maybe_gamepad_resource.has_value() && maybe_window_resource.has_value()) {
        // draw window showing gamepad stats
        auto& gui_window = maybe_window_resource.value().get();
        auto& connected_gamepads = maybe_gamepad_resource.value().get();

        gui_window.register_window("GLFW Gamepads ", [&](megamol::gui::AbstractWindow::BasicConfig& window_config) {
            for (auto& gamepad : connected_gamepads.gamepads) {
                window_config.flags = ImGuiWindowFlags_AlwaysAutoResize;

                ImGui::Text(gamepad.get().name.c_str());

                int i = 0;
                for (auto axis : gamepad.get().axes) {
                    ImGui::Text(("Axis " + std::to_string(i) + ": " + std::to_string(axis)).c_str());
                    ImGui::SameLine();
                    i++;
                }
                ImGui::NewLine();

                i = 0;
                for (auto button : gamepad.get().buttons) {
                    ImGui::Text(("Button " + std::to_string(i) + ": " + std::to_string(button)).c_str());
                    ImGui::SameLine();
                    i++;
                }
                ImGui::NewLine();

                i = 0;
                for (auto hat : gamepad.get().hats) {
                    ImGui::Text(("Hat " + std::to_string(i) + ": " + std::to_string(hat)).c_str());
                    ImGui::SameLine();
                    i++;
                }
                ImGui::NewLine();

                ImGui::Separator();
            }
        });
    }
}


} // namespace frontend
} // namespace megamol
