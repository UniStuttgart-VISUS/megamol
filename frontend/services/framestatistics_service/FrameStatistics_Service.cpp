/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

// search/replace FrameStatistics_Service with your class name
// you should also delete the FAQ comments in these template files after you read and understood them
#include "FrameStatistics_Service.hpp"

#include <chrono>
#include <numeric>
#include <sstream>

#include "LuaApiResource.h"
#include "mmcore/LuaAPI.h"
#include "mmcore/utility/Timestamp.h"
#include "mmcore/utility/log/Log.h"

// local logging wrapper for your convenience until central MegaMol logger established
static void log(const char* text) {
    const std::string msg = "FrameStatistics_Service: " + std::string(text);
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}

static void log(std::string text) {
    log(text.c_str());
}

namespace megamol::frontend {

FrameStatistics_Service::FrameStatistics_Service() {
    m_frame_times_micro.fill(0);
}

FrameStatistics_Service::~FrameStatistics_Service() {}

bool FrameStatistics_Service::init(void* configPtr) {
    if (configPtr == nullptr)
        return false;

    return init(*static_cast<Config*>(configPtr));
}

bool FrameStatistics_Service::init(const Config& config) {

    this->m_requestedResourcesNames = {//"IOpenGL_Context", // for GL-specific measures?
        frontend_resources::LuaAPI_Req_Name};

    m_program_start_time = std::chrono::steady_clock::time_point::clock::now();

    // the first frame is bogus, sorry.
    m_frame_start_time = std::chrono::steady_clock::time_point::clock::now();

    frame_stats_cb_.mark_frame = std::bind(&FrameStatistics_Service::mark_frame_cb, this);

    log("initialized successfully");
    return true;
}

void FrameStatistics_Service::close() {}

std::vector<FrontendResource>& FrameStatistics_Service::getProvidedResources() {
    m_providedResourceReferences = {{frontend_resources::FrameStatistics_Req_Name, m_statistics},
        {frontend_resources::FrameStatsCallbacks_Req_Name, frame_stats_cb_}};

    return m_providedResourceReferences;
}

const std::vector<std::string> FrameStatistics_Service::getRequestedResourceNames() const {
    return m_requestedResourcesNames;
}

void FrameStatistics_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    this->m_requestedResourceReferences = resources;
    fill_lua_callbacks();
}

void FrameStatistics_Service::updateProvidedResources() {}

void FrameStatistics_Service::digestChangedRequestedResources() {}

void FrameStatistics_Service::resetProvidedResources() {}

void FrameStatistics_Service::preGraphRender() {}

void FrameStatistics_Service::postGraphRender() {}

void FrameStatistics_Service::mark_frame_cb() {
    auto now = std::chrono::steady_clock::time_point::clock::now();

    ++m_statistics.rendered_frames_count;
    core::utility::log::Log::DefaultLog.WriteInfo("rendered_frames_count is now %u", m_statistics.rendered_frames_count);

    m_statistics.elapsed_program_time_milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - m_program_start_time);

    m_statistics.last_rendered_frame_time_microseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(now - m_frame_start_time);

    const auto oldest_sample = m_frame_times_micro[m_ring_buffer_ptr];
    m_frame_times_micro[m_ring_buffer_ptr] = m_statistics.last_rendered_frame_time_microseconds.count();
    m_ring_buffer_ptr = (m_ring_buffer_ptr + 1) % m_frame_times_micro.size();

    // this is expensive. I want to be rid of it.
    //m_statistics.last_averaged_mspf = std::accumulate(m_frame_times_micro.begin(), m_frame_times_micro.end(), 0) /
    //                                  static_cast<double>(m_frame_times_micro.size()) / 1000.0;
    //core::utility::log::Log::DefaultLog.WriteInfo("old mspf: %lf", m_statistics.last_averaged_mspf);

    // drop the oldest, add the newest: moving sum
    m_frame_times_sum = m_frame_times_sum + m_statistics.last_rendered_frame_time_microseconds.count() - oldest_sample;
    // which do we want, the one where an expensive first frame creates a visible perf dip that is not mitigated by "zero cost" previous frames...
    m_statistics.last_averaged_mspf =
        static_cast<double>(m_frame_times_sum) /
        static_cast<double>(std::min(m_frame_times_micro.size(), m_statistics.rendered_frames_count)) / 1000.0;
    // ... or the old implementation, that starts averaging over the whole buffer from the beginning
    //m_statistics.last_averaged_mspf = static_cast<double>(m_frame_times_sum) /
    //                                  static_cast<double>(m_frame_times_micro.size()) / 1000.0;
    //core::utility::log::Log::DefaultLog.WriteInfo("new mspf: %lf", m_statistics.last_averaged_mspf);

    m_statistics.last_averaged_fps = 1000.0 / m_statistics.last_averaged_mspf;
    m_frame_start_time = now;
}

void FrameStatistics_Service::fill_lua_callbacks() {
    auto& luaApi = m_requestedResourceReferences[0].getResource<core::LuaAPI*>();

    luaApi->RegisterCallback("mmGetTimeStamp", "(void)\n\tReturns a timestamp in ISO format.", [&]() -> std::string {
        auto const tp = std::chrono::system_clock::now();
        auto const timestamp = core::utility::serialize_timestamp(tp);
        return timestamp;
    });
}

} // namespace megamol::frontend
