/*
 * FrameStatistics_Service.cpp
 *
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

// search/replace FrameStatistics_Service with your class name
// you should also delete the FAQ comments in these template files after you read and understood them
#include "FrameStatistics_Service.hpp"

#include <numeric>


// local logging wrapper for your convenience until central MegaMol logger established
#include "mmcore/utility/log/Log.h"
static void log(const char* text) {
    const std::string msg = "FrameStatistics_Service: " + std::string(text);
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}
static void log(std::string text) {
    log(text.c_str());
}

namespace megamol {
namespace frontend {

FrameStatistics_Service::FrameStatistics_Service() {}

FrameStatistics_Service::~FrameStatistics_Service() {}

bool FrameStatistics_Service::init(void* configPtr) {
    if (configPtr == nullptr)
        return false;

    return init(*static_cast<Config*>(configPtr));
}

bool FrameStatistics_Service::init(const Config& config) {

    this->m_requestedResourcesNames = {
        //"IOpenGL_Context", // for GL-specific measures?
    };

    m_program_start_time = std::chrono::high_resolution_clock::now();

    log("initialized successfully");
    return true;
}

void FrameStatistics_Service::close() {}

std::vector<FrontendResource>& FrameStatistics_Service::getProvidedResources() {
    m_providedResourceReferences = {{"FrameStatistics", m_statistics}};

    return m_providedResourceReferences;
}

const std::vector<std::string> FrameStatistics_Service::getRequestedResourceNames() const {
    return m_requestedResourcesNames;
}

void FrameStatistics_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    this->m_requestedResourceReferences = resources;
}

void FrameStatistics_Service::updateProvidedResources() {
    start_frame();
}

void FrameStatistics_Service::digestChangedRequestedResources() {}

void FrameStatistics_Service::resetProvidedResources() {
    finish_frame();
}

void FrameStatistics_Service::preGraphRender() {}

void FrameStatistics_Service::postGraphRender() {}

// TODO: maybe port FPS Counter from
// #include "vislib/graphics/FpsCounter.h"
void FrameStatistics_Service::start_frame() {
    m_frame_start_time = std::chrono::high_resolution_clock::now();
}

void FrameStatistics_Service::finish_frame() {
    auto now = std::chrono::high_resolution_clock::now();

    m_statistics.rendered_frames_count++;

    m_statistics.elapsed_program_time_seconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - m_program_start_time).count() /
        static_cast<double>(1000);

    auto last_frame_till_now_micro =
        std::chrono::duration_cast<std::chrono::microseconds>(now - m_frame_start_time).count();

    m_statistics.last_rendered_frame_time_milliseconds = last_frame_till_now_micro / static_cast<double>(1000);

    m_frame_times_micro[m_ring_buffer_ptr] = last_frame_till_now_micro;
    m_ring_buffer_ptr = (m_ring_buffer_ptr + 1) % m_frame_times_micro.size();

    m_statistics.last_averaged_mspf = std::accumulate(m_frame_times_micro.begin(), m_frame_times_micro.end(), 0) /
                                      m_frame_times_micro.size() / static_cast<double>(1000);
    m_statistics.last_averaged_fps = 1000.0 / m_statistics.last_averaged_mspf;
}

} // namespace frontend
} // namespace megamol
