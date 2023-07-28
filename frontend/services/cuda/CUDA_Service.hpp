/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#ifdef MM_CUDA_ENABLED

#include "AbstractFrontendService.hpp"
#include "CUDA_Context.h"

namespace megamol::frontend {
class CUDA_Service : public AbstractFrontendService {
public:
    CUDA_Service() = default;

    ~CUDA_Service() override = default;

    std::string serviceName() const override {
        return "CUDA_Service";
    };

    bool init(void* configPtr) override;
    void close() override;

    void updateProvidedResources() override{};
    void digestChangedRequestedResources() override{};
    void resetProvidedResources() override{};

    void preGraphRender() override{};
    void postGraphRender() override{};

    std::vector<FrontendResource>& getProvidedResources() override {
        return resourceReferences_;
    };

    const std::vector<std::string> getRequestedResourceNames() const override {
        return {};
    };
    void setRequestedResources(std::vector<FrontendResource> resources) override{};

private:
    frontend_resources::CUDA_Context ctx_;

    std::vector<FrontendResource> resourceReferences_;
};
} // namespace megamol::frontend

#endif
