/*
 * Screenshot_Service.hpp
 *
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <fstream>
#include <sstream>

#include "AbstractFrontendService.hpp"
#include "FrameStatistics.h"
#include "PerformanceManager.h"

namespace megamol::frontend {

class Profiling_Service final : public AbstractFrontendService {
public:
    struct Config {
        std::string log_file;
        uint32_t flush_frequency;
        bool autostart_profiling;
        bool include_graph_events;
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
    void setRequestedResources(std::vector<FrontendResource> resources) override;

private:
    void fill_lua_callbacks();
    void log_graph_event(std::string const& parent, std::string const& name, std::string const& comment);

    std::vector<FrontendResource> _providedResourceReferences;
    std::vector<std::string> _requestedResourcesNames;
    std::vector<FrontendResource> _requestedResourcesReferences;

    frontend_resources::PerformanceManager _perf_man;
    uint32_t flush_frequency = 0;
    std::ofstream log_file;
    std::stringstream log_buffer;
    bool include_graph_events = false;
    frontend_resources::ProfilingLoggingStatus profiling_logging;
};

} // namespace megamol::frontend
