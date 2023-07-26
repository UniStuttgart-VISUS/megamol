/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <vector>

#include "AbstractFrontendService.hpp"

namespace megamol::frontend {

// provides abstraction to call AbstractFrontendService methods on a collection of services at once
class FrontendServiceCollection {
public:
    FrontendServiceCollection() = default;
    ~FrontendServiceCollection() = default;

    void add(AbstractFrontendService& service, void* service_config);

    bool init();
    void close();

    bool assignRequestedResources();
    std::vector<FrontendResource>& getProvidedResources();

    void updateProvidedResources();
    void digestChangedRequestedResources();
    void resetProvidedResources();

    void preGraphRender();
    void postGraphRender();

    bool shouldShutdown() const;

private:
    void sortServices();
    FrontendResource* findResource(std::string const& name);

    struct ServiceEntry {
        AbstractFrontendService* service = nullptr;
        void* service_config = nullptr;

        AbstractFrontendService const& get() const {
            return *service;
        }
        AbstractFrontendService& get() {
            return *service;
        }
    };

    std::vector<ServiceEntry> m_services;
    std::vector<FrontendResource> m_serviceResources;
};

} // namespace megamol::frontend
