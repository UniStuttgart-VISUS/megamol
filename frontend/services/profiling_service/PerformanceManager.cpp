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
    timer_config conf;
    conf.parent = parent_type::BUILTIN;
    conf.user_index = 0;
#ifdef MEGAMOL_USE_OPENGL
    conf.name = "FrameTimeGL";
    conf.api = query_api::OPENGL;
    whole_frame_gl = add_timer(std::make_unique<gl_timer>(conf));
#endif
    conf.name = "FrameTime";
    conf.api = query_api::CPU;
    whole_frame_cpu = add_timer(std::make_unique<cpu_timer>(conf));
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

timer_region& PerformanceManager::start_timer(handle_type h) {
    return timers[h]->start(current_frame);
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
    //core::utility::log::Log::DefaultLog.WriteInfo("PerformanceManager: starting frame %u", frame);
    //gl_timer::last_query = 0;
    // we never reset the global counter!
    //current_global_index = 0;
    if (first_frame) {
        whole_frame_cpu_region = &start_timer(whole_frame_cpu);
#ifdef MEGAMOL_USE_OPENGL
        whole_frame_gl_region = &start_timer(whole_frame_gl);
#endif
        first_frame = false;
    }
}

void PerformanceManager::collect_timer_and_append(Itimer* timer, frame_info& the_frame) {
    //timer->Itimer::collect();
    timer->collect(the_frame.frame);
    auto& tconf = timer->Itimer::get_conf();
    timer_entry e;
    e.handle = timer->Itimer::get_handle();
    e.user_index = tconf.basic_timer_config::user_index;
    e.parent = tconf.timer_config::parent;

    //core::utility::log::Log::DefaultLog.WriteInfo("PerformanceManager: pushing data for frame %u with %u regions",
    //    the_frame.frame, timer->Itimer::get_region_count());

    for (auto region = timer->regions_begin(); region != timer->regions_end(); ++region) {
        /*if (timer->Itimer::get_frame(region) != the_frame.frame) {
            throw std::runtime_error("PerformanceManager: frame inconsistency detected in region!");
        }*/

        if (!region->finished) {
            continue;
        }

        e.frame = region->frame;

        e.frame_index = region->frame_index;
        e.api = tconf.basic_timer_config::api;

        e.global_index = region->global_index;
        e.start = region->start;
        e.end = region->end;
        e.duration = time_point{region->end - region->start};
        the_frame.entries.push_back(e);
    }
    // we cannot wait for a timer to be re-started to remove the regions lying around there:
    // if it does not start, the regions will persist...
    timer->clear(the_frame.frame);
}

void PerformanceManager::endFrame(frame_type frame) {
    //core::utility::log::Log::DefaultLog.WriteInfo("PerformanceManager: ending frame %u", frame);
#ifdef MEGAMOL_USE_OPENGL
    whole_frame_gl_region->end_region();
#endif
    whole_frame_cpu_region->end_region();

    // and start again for next frame
    whole_frame_cpu_region = &timers[whole_frame_cpu]->start(current_frame + 1);
#ifdef MEGAMOL_USE_OPENGL
    whole_frame_gl_region = &timers[whole_frame_gl]->start(current_frame + 1);
#endif


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
                // we cannot check this anymore, sad.
                //if (timer->started) {
                //    // timer was not ended this frame, that is not nice
                //    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                //        "PerformanceManager: timer %s was not properly ended in frame %u",
                //        timer->get_conf().name.c_str(), current_frame);
                //    continue;
                //}
            }
        }

        frame_buffer.frame = current_frame;
        frame_buffer.entries.clear();

        for (auto& [key, timer] : timers) {
            collect_timer_and_append(timer.get(), frame_buffer);
        }

        std::sort(frame_buffer.entries.begin(), frame_buffer.entries.end(),
            [](const timer_entry& a, const timer_entry& b) { return a.global_index < b.global_index; });

        for (auto& subscriber : subscribers) {
            subscriber(frame_buffer);
        }
    }
}
} // namespace megamol::frontend_resources::performance
