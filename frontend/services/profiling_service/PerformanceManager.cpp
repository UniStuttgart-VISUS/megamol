/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "PerformanceManager.h"

#include <array>

#include "mmcore/Call.h"
#include "mmcore/Module.h"

#ifdef MEGAMOL_USE_OPENGL
#include <glad/gl.h>
#endif

namespace megamol::frontend_resources::performance {

PerformanceManager::PerformanceManager() {
#ifdef MEGAMOL_USE_OPENGL
    whole_frame_gl = add_timer("FrameTimeGL", query_api::OPENGL);
#endif
    whole_frame_cpu = add_timer("FrameTime", query_api::CPU);
}

handle_vector PerformanceManager::add_timers(megamol::core::Module* m, std::vector<basic_timer_config> timer_list) {
    handle_vector ret;
    timer_config conf;
    conf.parent_pointer = m;
    conf.parent = parent_type::USER_REGION;
    for (const auto& btc : timer_list) {
        conf.api = btc.api;
        conf.name = btc.name;
        switch (conf.api) {
        case query_api::CPU: {
            ret.push_back(add_timer(std::make_unique<cpu_timer>(conf)));
            break;
        }
        case query_api::OPENGL: {
            ret.push_back(add_timer(std::make_unique<gl_timer>(conf)));
            break;
        }
        }
    }
    return ret;
}

handle_vector PerformanceManager::add_timers(megamol::core::Call* c, query_api api) {
    handle_vector ret;
    const auto caps = c->GetCapabilities();
    timer_config conf;
    conf.parent_pointer = c;
    conf.parent = parent_type::CALL;
    conf.api = api;
    for (auto i = 0; i < c->GetCallbackCount(); ++i) {
        conf.name = c->GetCallbackName(i);
        conf.user_index = i;
        switch (api) {
        case query_api::CPU:
            ret.push_back(add_timer(std::make_unique<cpu_timer>(conf)));
            break;
        case query_api::OPENGL:
            ret.push_back(add_timer(std::make_unique<gl_timer>(conf)));
            break;
        }
    }
    return ret;
}

handle_type PerformanceManager::add_timer(std::string name, query_api api) {
    handle_type ret;
    timer_config conf;
    conf.parent = parent_type::BUILTIN;
    conf.api = api;
    conf.name = name;
    conf.user_index = 0;
    switch (api) {
    case query_api::CPU:
        return add_timer(std::make_unique<cpu_timer>(conf));
    case query_api::OPENGL:
        return add_timer(std::make_unique<gl_timer>(conf));
    }
    return 0;
}

void PerformanceManager::remove_timers(handle_vector handles) {
    for (auto handle : handles) {
        timers.erase(handle);
    }
    handle_holes.insert(handle_holes.end(), handles.begin(), handles.end());
}

std::string PerformanceManager::lookup_parent(handle_type h) {
    const auto& conf = timers[h]->get_conf();
    return Itimer::parent_name(conf);
}

void* PerformanceManager::lookup_parent_pointer(handle_type h) {
    return timers[h]->get_conf().parent_pointer;
}

parent_type PerformanceManager::lookup_parent_type(handle_type h) {
    return timers[h]->get_conf().parent;
}

std::string PerformanceManager::lookup_name(handle_type h) {
    return timers[h]->get_conf().name;
}

const timer_config PerformanceManager::lookup_config(handle_type h) {
    return timers[h]->get_conf();
}

handle_vector PerformanceManager::lookup_timers(void* parent) {
    handle_vector vec;
    for (auto& t : timers) {
        if (t.second->get_conf().parent_pointer == parent) {
            vec.push_back(t.second->get_handle());
        }
    }
    return vec;
}

void PerformanceManager::set_transient_comment(handle_type h, std::string comment) {
    if (timers.find(h) != timers.end()) {
        timers[h]->set_comment(comment);
    } else {
        core::utility::log::Log::DefaultLog.WriteError("PerformanceManager: cannot find timer with handle %u", h);
    }
}

void PerformanceManager::subscribe_to_updates(update_callback cb) {
    subscribers.push_back(cb);
}

void PerformanceManager::start_timer(handle_type h) {
    timers[h]->start(current_frame);
}

void PerformanceManager::stop_timer(handle_type h) {
    timers[h]->end();
}

handle_type PerformanceManager::add_timer(std::unique_ptr<Itimer> t) {
    handle_type my_handle = 0;
    if (!handle_holes.empty()) {
        my_handle = handle_holes.back();
        handle_holes.pop_back();
    } else {
        my_handle = current_handle;
        current_handle++;
    }
    t->h = my_handle;
    auto pair = std::make_pair(my_handle, std::move(t));
    timers.insert(std::move(pair));
    return my_handle;
}

void PerformanceManager::startFrame(frame_type frame) {
    current_frame = frame;
    core::utility::log::Log::DefaultLog.WriteInfo("PerformanceManager: starting frame %u", frame);
    //gl_timer::last_query = 0;
    // we never reset the global counter!
    //current_global_index = 0;
    start_timer(whole_frame_cpu);
#ifdef MEGAMOL_USE_OPENGL
    start_timer(whole_frame_gl);
#endif
}

void PerformanceManager::collect_timer_and_append(Itimer* timer, frame_info& the_frame) {
    //timer->Itimer::collect();
    timer->collect(the_frame.frame);
    auto& tconf = timer->Itimer::get_conf();
    timer_entry e;
    e.handle = timer->Itimer::get_handle();
    e.user_index = tconf.basic_timer_config::user_index;
    e.parent = tconf.timer_config::parent;

    core::utility::log::Log::DefaultLog.WriteInfo(
        "PerformanceManager: pushing data for frame %u with %u regions", the_frame.frame, timer->Itimer::get_region_count());

    for (uint32_t region = 0; region < timer->Itimer::get_region_count(); ++region) {
        /*if (timer->Itimer::get_frame(region) != the_frame.frame) {
            throw std::runtime_error("PerformanceManager: frame inconsistency detected in region!");
        }*/

        if (!timer->Itimer::get_ready(region))
            continue;

        e.frame = timer->Itimer::get_frame(region);

        e.frame_index = region;
        e.api = tconf.basic_timer_config::api;

        e.global_index = timer->Itimer::get_global_index(region);
        e.start = timer->Itimer::get_start(region);
        e.end = timer->Itimer::get_end(region);
        e.duration = time_point{timer->Itimer::get_end(region) - timer->Itimer::get_start(region)};
        the_frame.entries.push_back(e);
    }
    // we cannot wait for a timer to be re-started to remove the regions lying around there:
    // if it does not start, the regions will persist...
    timer->clear(the_frame.frame);
}

void PerformanceManager::endFrame(frame_type frame) {
    core::utility::log::Log::DefaultLog.WriteInfo("PerformanceManager: ending frame %u", frame);
#ifdef MEGAMOL_USE_OPENGL
    stop_timer(whole_frame_gl);
#endif
    stop_timer(whole_frame_cpu);

    if (current_frame != frame) {
        throw std::runtime_error(("PerformanceManager: ending frame " + std::to_string(frame) +
                                  " does not fit the frame we are in: " + std::to_string(current_frame))
                                     .c_str());
    }

    if (!subscribers.empty()) {
        // consistency check for current frame
        for (auto& [key, timer] : timers) {
            if (timer->get_start_frame() != current_frame) {
                // timer did not start this frame
                continue;
            } else {
                if (timer->started) {
                    // timer was not ended this frame, that is not nice
                    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                        "PerformanceManager: timer %s was not properly ended in frame %u",
                        timer->get_conf().name.c_str(), current_frame);
                    continue;
                }
            }
        }

        //frame_info& previous_frame = frame_double_buffer[(current_frame + 1) % 2];
        frame_info& this_frame = frame_double_buffer[0];

        this_frame.frame = current_frame;
        this_frame.entries.clear();

        for (auto& [key, timer] : timers) {
            if (timer->conf.api == query_api::OPENGL) {
                // get GPU stuff from previous frame
                collect_timer_and_append(timer.get(), this_frame);
            } else {
                // currently equivalent to if (timer->conf.api == query_api::CPU)
                // get CPU stuff from current frame
                collect_timer_and_append(timer.get(), this_frame);
            }
        }

        // report previous frame, because only that is complete
        // this causes a 1-frame delay in the GUI but that should be OK
        std::sort(this_frame.entries.begin(), this_frame.entries.end(),
            [](const timer_entry& a, const timer_entry& b) { return a.global_index < b.global_index; });

         /*megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "PerformanceManager: in frame %u report on frame %u while current frame is %u", frame, previous_frame.frame,
            current_frame);*/

        for (auto& subscriber : subscribers) {
            subscriber(this_frame);
        }
    }
}
} // namespace megamol::frontend_resources::performance
