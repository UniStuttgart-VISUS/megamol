/*
 * Screenshot_Service.hpp
 *
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractFrontendService.hpp"
#include "mmcore/CallProfiling.h"
#include "PerformanceManager.h"

namespace megamol {
namespace frontend {

class Profiling_Service final : public AbstractFrontendService {
public:
    std::string serviceName() const override { return "Profiling_Service"; }
    bool init(void* configPtr) override;
    void close() override {}
    void updateProvidedResources() override { _pman.startFrame(); }
    void digestChangedRequestedResources() override {}

    void resetProvidedResources() override {
        core::CallProfiling::CollectGPUPerformance();
        _pman.endFrame();
        // TODO append performance log file
    }

    void preGraphRender() override {}
    void postGraphRender() override {}
    std::vector<FrontendResource>& getProvidedResources() override { return _providedResourceReferences; }
    const std::vector<std::string> getRequestedResourceNames() const override { return _requestedResourcesNames; }
    void setRequestedResources(std::vector<FrontendResource> resources) override {}

    std::vector<FrontendResource> _providedResourceReferences;
    std::vector<std::string> _requestedResourcesNames;

    frontend_resources::PerformanceManager _pman;
};

} // namespace frontend
} // namespace megamol
