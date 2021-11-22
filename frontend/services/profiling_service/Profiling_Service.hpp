/*
 * Screenshot_Service.hpp
 *
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "AbstractFrontendService.hpp"
#include "mmcore/CallProfiling.h"

namespace megamol {
namespace frontend {

class Profiling_Service final : public AbstractFrontendService {
public:
    std::string serviceName() const override {
        return "Profiling_Service";
    }
    bool init(void* configPtr) override {
        return true;
    }
    void close() override {}
    void updateProvidedResources() override {}
    void digestChangedRequestedResources() override {}

    void resetProvidedResources() override {
        core::CallProfiling::CollectGPUPerformance();
    }

    void preGraphRender() override {}
    void postGraphRender() override {}
    std::vector<FrontendResource>& getProvidedResources() override {
        return m_providedResourceReferences;
    }
    const std::vector<std::string> getRequestedResourceNames() const override {
        return m_requestedResourcesNames;
    }
    void setRequestedResources(std::vector<FrontendResource> resources) override {}

    std::vector<FrontendResource> m_providedResourceReferences;
    std::vector<std::string> m_requestedResourcesNames;
};

} // namespace frontend
} // namespace megamol
