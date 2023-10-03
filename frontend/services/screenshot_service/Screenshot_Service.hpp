/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "AbstractFrontendService.hpp"

// ImageData struct and interfaces for screenshot sources/writers
#include "Screenshots.h"

#include "RuntimeInfo.h"

namespace megamol::frontend {

class Screenshot_Service final : public AbstractFrontendService {
public:
    struct Config {
        bool show_privacy_note;
    };

    std::string serviceName() const override {
        return "Screenshot_Service";
    }

    Screenshot_Service();
    ~Screenshot_Service() override;

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
    // you inherit the following functions
    //
    // int setPriority(const int p) // priority initially 0
    // int getPriority() const;
    //
    // bool shouldShutdown() const; // shutdown initially false
    // void setShutdown(const bool s = true);

    static unsigned char default_alpha_value;

private:
    megamol::frontend_resources::GLScreenshotSource m_frontbufferSource_resource;
    megamol::frontend_resources::ScreenshotImageDataToPNGWriter m_toFileWriter_resource;

    std::function<bool(std::filesystem::path const&)> m_frontbufferToPNG_trigger;
    std::function<bool(megamol::frontend_resources::ImageWrapper const&, std::filesystem::path const&)>
        m_imagewrapperToPNG_trigger;

    std::vector<FrontendResource> m_providedResourceReferences;
    std::vector<std::string> m_requestedResourcesNames;
    std::vector<FrontendResource> m_requestedResourceReferences;

    megamol::frontend_resources::RuntimeInfo const* ri_;
};

} // namespace megamol::frontend
