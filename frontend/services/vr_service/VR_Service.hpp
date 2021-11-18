/*
 * VR_Service.hpp
 *
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractFrontendService.hpp"

namespace megamol {
namespace frontend {

// VR Service monitors Image Presentation Entry Points (Views, which output graph rendering results)
// and injects them with VR/AR data as it sees fit.
// this means we may take control over View Camera state
// (resolution, tiling, camera pose/projection) and inject our own data to achieve stereo rendering.
// we may also manipulate entry points set by the graph,
// delete or clone them to get the number of output renderings of the graph we need.

class VR_Service final : public AbstractFrontendService {
public:
    struct Config {
        enum class Mode {
            Off,
        };

        Mode mode = Mode::Off;
    };

    std::string serviceName() const override {
        return "VR_Service";
    }

    VR_Service();
    ~VR_Service();

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
    //
    // int setPriority(const int p) // priority initially 0
    // int getPriority() const;
    //
    // bool shouldShutdown() const; // shutdown initially false
    // void setShutdown(const bool s = true);

private:
    std::vector<FrontendResource> m_providedResourceReferences;
    std::vector<std::string> m_requestedResourcesNames;
    std::vector<FrontendResource> m_requestedResourceReferences;
};

} // namespace frontend
} // namespace megamol
