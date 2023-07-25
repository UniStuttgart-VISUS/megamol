/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "AbstractFrontendService.hpp"

#include "CommandRegistry.h"

namespace megamol::frontend {
class Command_Service : public AbstractFrontendService {
public:
    Command_Service() = default;

    ~Command_Service() override = default;

    std::string serviceName() const override {
        return "Command_Service";
    };

    bool init(void* configPtr) override;
    void close() override;

    void updateProvidedResources() override{};
    void digestChangedRequestedResources() override;
    void resetProvidedResources() override{};

    void preGraphRender() override{};
    void postGraphRender() override{};

    std::vector<FrontendResource>& getProvidedResources() override {
        return providedResourceReferences;
    };

    const std::vector<std::string> getRequestedResourceNames() const override {
        return requestedResourceNames;
    };
    void setRequestedResources(std::vector<FrontendResource> resources) override;

private:
    frontend_resources::CommandRegistry commands;

    std::vector<FrontendResource> providedResourceReferences;
    std::vector<FrontendResource> requestedResourceReferences;
    std::vector<std::string> requestedResourceNames;
};
} // namespace megamol::frontend
