/*
 * Screenshot_Service.hpp
 *
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractFrontendService.hpp"

// ImageData struct and interfaces for screenshot sources/writers
#include "Screenshots.h"

namespace megamol {
namespace frontend {

// for detailed service API documentation see Template_Service.{hpp,cpp}
// 
// 
class Screenshot_Service final : public AbstractFrontendService {
public:

    struct Config {
    };

    std::string serviceName() const override { return "Screenshot_Service"; }

    Screenshot_Service();
    ~Screenshot_Service();

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

private:
    megamol::frontend_resources::GLScreenshotSource m_frontbufferSource_resource;
    megamol::frontend_resources::ScreenshotImageDataToPNGWriter m_toFileWriter_resource;

    std::function<bool(std::string const&)> m_frontbufferToPNG_trigger;

    std::vector<FrontendResource> m_providedResourceReferences;
    std::vector<std::string> m_requestedResourcesNames;
    std::vector<FrontendResource> m_requestedResourceReferences;
};

} // namespace frontend
} // namespace megamol
