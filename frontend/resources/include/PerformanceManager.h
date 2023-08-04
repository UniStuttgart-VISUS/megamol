/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "Timers.h"

#include <chrono>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace megamol {
namespace core {
class Call;
class Module;
} // namespace core
namespace frontend {
class Profiling_Service;
} // namespace frontend
} // namespace megamol

namespace megamol::frontend_resources::performance {

static std::string Performance_Logging_Status_Req_Name = "ProfilingLoggingStatus";

struct ProfilingLoggingStatus {
    bool active = true;
};

static std::string PerformanceManager_Req_Name = "PerformanceManager";

// this thing must only exist ONCE.
class PerformanceManager {
public:
    PerformanceManager();

    ~PerformanceManager() = default;

    using update_callback = std::function<void(const frame_info&)>;

    // names and API defined explicitly for modules
    handle_vector add_timers(megamol::core::Module* m, std::vector<basic_timer_config> timer_list);

    // names derived from callbacks
    handle_vector add_timers(megamol::core::Call* c, query_api api);

    // explicit name
    handle_type add_timer(std::string name, query_api api);

    void remove_timers(handle_vector handles);

    // hint: this is not for free, so don't call this all the time
    std::string lookup_parent(handle_type h);

    // hint: this is not for free, so don't call this all the time
    void* lookup_parent_pointer(handle_type h);

    // hint: this is not for free, so don't call this all the time
    parent_type lookup_parent_type(handle_type h);

    // hint: this is not for free, so don't call this all the time
    std::string lookup_name(handle_type h);

    // hint: this is not for free, so don't call this all the time
    const timer_config lookup_config(handle_type h);

    // hint: this is not for free, so don't call this all the time
    handle_vector lookup_timers(void* parent);

    void set_transient_comment(handle_type h, std::string comment);

    void subscribe_to_updates(update_callback cb);

    void start_timer(handle_type h);
    void stop_timer(handle_type h);

private:
    friend class frontend::Profiling_Service;

    handle_type add_timer(std::unique_ptr<Itimer> t);

    void startFrame(frame_type frame);
    static void collect_timer_and_append(Itimer* timer, frame_info& the_frame);

    void endFrame(frame_type frame);

    handle_type current_handle = 0;
    std::vector<handle_type> handle_holes;
    std::unordered_map<handle_type, std::unique_ptr<Itimer>> timers;
    frame_type current_frame = 0;
    std::vector<update_callback> subscribers;

    std::array<frame_info, 2> frame_double_buffer;

#ifdef MEGAMOL_USE_OPENGL
    handle_type whole_frame_gl;
#endif
    handle_type whole_frame_cpu;
};

} // namespace megamol::frontend_resources::performance
