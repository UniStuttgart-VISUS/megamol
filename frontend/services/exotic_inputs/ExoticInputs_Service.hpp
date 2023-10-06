/*
 * ExoticInputs_Service.hpp
 *
 * Copyright (C) 2022 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractFrontendService.hpp"

#include <map>
#include <optional>

namespace megamol {
namespace frontend {

// The Exotic Inputs Service is supposed to implement the injection of raw input device state (e.g.gamepads via GLFW)
// into the MegaMol Graph by matching input devices to graph modules which can be controlled in some usefull way using that device
// So this ervice looks up present input device resources, or manages input devices on his own, and knows how to connect them to graph modules
// For example, gamepads may be used to control View3D Cameras or positions of Clipping Planes in space
class ExoticInputs_Service final : public AbstractFrontendService {
public:
    struct Config {};

    std::string serviceName() const override {
        return "ExoticInputs_Service";
    }

    ExoticInputs_Service();
    ~ExoticInputs_Service();

    bool init(const Config& config);
    bool init(void* configPtr) override;
    void close() override;

    std::vector<FrontendResource>& getProvidedResources() override;
    const std::vector<std::string> getRequestedResourceNames() const override;
    void setRequestedResources(std::vector<FrontendResource> resources) override;

    void updateProvidedResources() override;
    void digestChangedRequestedResources() override;
    void resetProvidedResources() override;
    void preGraphRender() override;
    void postGraphRender() override;

    // from AbstractFrontendService
    // int setPriority(const int p) // priority initially 0
    // int getPriority() const;
    // bool shouldShutdown() const; // shutdown initially false
    // void setShutdown(const bool s = true);

private:
    std::vector<FrontendResource> m_providedResourceReferences;
    std::vector<std::string> m_requestedResourcesNames;
    std::vector<FrontendResource> m_requestedResourceReferences;

    void gamepad_window() const;

    enum class PoseManipulator {
        Arcball = 0,
        //Turntable,
        FPS,
        COUNT,
    };
    PoseManipulator m_manipulation_mode = PoseManipulator::Arcball;

    float m_axis_threshold = 0.05f;

    // all known view3d entry points
    std::map<std::string, void*> m_view3d_modules;

    // the entry point currently controlled
    std::optional<std::string> m_controlled_view3d = std::nullopt;
};

} // namespace frontend
} // namespace megamol
