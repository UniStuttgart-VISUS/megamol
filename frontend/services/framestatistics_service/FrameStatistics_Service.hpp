/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <array>
#include <chrono>

#include "AbstractFrontendService.hpp"
#include "FrameStatistics.h"
#include "FrameStatsCallbacks.h"

namespace megamol::frontend {

class FrameStatistics_Service final : public AbstractFrontendService {
public:
    struct Config {};

    std::string serviceName() const override {
        return "FrameStatistics_Service";
    }

    FrameStatistics_Service();
    ~FrameStatistics_Service() override;

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
    // you inherit the following functions that manage priority of your service and shutdown requests to terminate the program
    // you and others may use those functions, but you will not override them
    // priority indicates the order in which services get their callbacks called, i.e. this is the sorting of the vector that holds all services
    // lower priority numbers get called before the bigger ones. for close() and postGraphRender() services get called in the reverse order,
    // i.e. this works like construction and destruction order of objects in a c++
    //
    // int setPriority(const int p) // priority initially 0
    // int getPriority() const;

    // your service can signal to the program that a shutdown request has been received.
    // call setShutdown() to set your shutdown status to true, this is best done in your updateProvidedResources() or digestChangedRequestedResources().
    // if a servie signals a shutdown the system calls close() on all services in reverse priority order, then program execution terminates.
    //
    // bool shouldShutdown() const; // shutdown initially false
    // void setShutdown(const bool s = true);

private:
    void mark_frame_cb();
    frontend_resources::FrameStatsCallbacks frame_stats_cb_;

    megamol::frontend_resources::FrameStatistics m_statistics;

    std::chrono::steady_clock::time_point m_program_start_time;
    std::chrono::steady_clock::time_point m_frame_start_time;

    std::array<long long, 30> m_frame_times_micro;
    long long m_frame_times_sum = 0;
    unsigned int m_ring_buffer_ptr = 0;

    void fill_lua_callbacks();

    std::vector<FrontendResource> m_providedResourceReferences;
    std::vector<std::string> m_requestedResourcesNames;
    std::vector<FrontendResource> m_requestedResourceReferences;
};

} // namespace megamol::frontend
