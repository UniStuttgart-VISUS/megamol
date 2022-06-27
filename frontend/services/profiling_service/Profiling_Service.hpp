/*
 * Screenshot_Service.hpp
 *
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <fstream>
#include <random>
#include <tuple>

#include "AbstractFrontendService.hpp"
#include "PerformanceManager.h"

namespace megamol {
namespace frontend {

class Profiling_Service final : public AbstractFrontendService {
public:
    struct Config {
        std::string log_file;
    };

    std::string serviceName() const override {
        return "Profiling_Service";
    }
    bool init(void* configPtr) override;
    void close() override;
    void updateProvidedResources() override;
    void digestChangedRequestedResources() override {}

    void resetProvidedResources() override;

    void preGraphRender() override {}
    void postGraphRender() override {}
    std::vector<FrontendResource>& getProvidedResources() override {
        return _providedResourceReferences;
    }
    const std::vector<std::string> getRequestedResourceNames() const override {
        return _requestedResourcesNames;
    }
    void setRequestedResources(std::vector<FrontendResource> resources) override {
        _requestedResourcesReferences = resources;
    }

    std::vector<FrontendResource> _providedResourceReferences;
    std::vector<std::string> _requestedResourcesNames;
    std::vector<FrontendResource> _requestedResourcesReferences;

    megamol::frontend_resources::PerformanceManager _perf_man;
    std::ofstream log_file;

private:
    void fill_lua_callbacks();

    void notify_timer_queries();

    std::unordered_map<int64_t, std::tuple<int, int64_t, int64_t>> timer_map_;

    std::mt19937_64 rng_;

    std::uniform_int_distribution<int64_t> distro_;
};

} // namespace frontend
} // namespace megamol
