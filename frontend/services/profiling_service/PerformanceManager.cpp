#include "PerformanceManager.h"

#include "mmcore/Call.h"
#include "mmcore/Module.h"
#include <array>

#ifdef MEGAMOL_USE_OPENGL
#include "glad/gl.h"
#endif

namespace megamol::frontend_resources {

bool PerformanceManager::Itimer::start(frame_type frame) {
    auto new_frame = false;
    if (frame != start_frame) {
        new_frame = true;
        regions.clear();
    }
    if (!started) {
        started = true;
        start_frame = frame;
    } else {
        throw std::runtime_error(
            ("timer: region " + parent_name(conf) + "::" + conf.name + " needs to be ended before being started")
                .c_str());
    }
    return new_frame;
}

void PerformanceManager::Itimer::end() {
    if (!started) {
        throw std::runtime_error(
            ("cpu_timer: region " + parent_name(conf) + "::" + conf.name + " needs to be started before being ended")
                .c_str());
    }
    started = false;
}

bool PerformanceManager::cpu_timer::start(frame_type frame) {
    const auto ret = Itimer::start(frame);
    last_start = time_point::clock::now();
    return ret;
}

void PerformanceManager::cpu_timer::end() {
    Itimer::end();
    auto end = time_point::clock::now();
    regions.emplace_back(std::make_pair(last_start, end));
}

PerformanceManager::gl_timer::~gl_timer() {
#ifdef MEGAMOL_USE_OPENGL
    for (auto& q_pair : query_ids) {
        glDeleteQueries(1, &q_pair.first);
        glDeleteQueries(1, &q_pair.second);
    }
#endif
}

bool PerformanceManager::gl_timer::start(frame_type frame) {
    const auto new_frame = Itimer::start(frame);
    if (new_frame) {
        query_index = 0;
    }
    last_query = assert_query(query_index).first;
#ifdef MEGAMOL_USE_OPENGL
    glQueryCounter(last_query, GL_TIMESTAMP);
#endif
    return new_frame;
}

void PerformanceManager::gl_timer::end() {
    Itimer::end();
    last_query = assert_query(query_index).second;
#ifdef MEGAMOL_USE_OPENGL
    glQueryCounter(last_query, GL_TIMESTAMP);
#endif
    query_index++;
}

void PerformanceManager::gl_timer::collect() {
#ifdef MEGAMOL_USE_OPENGL
    GLuint64 start_time, end_time;
    for (uint32_t index = 0; index < query_index; ++index) {
        const auto& [start, end] = query_ids[index];
        glGetQueryObjectui64v(start, GL_QUERY_RESULT, &start_time);
        glGetQueryObjectui64v(end, GL_QUERY_RESULT, &end_time);
        regions.emplace_back(std::make_pair(
            time_point{std::chrono::nanoseconds(start_time)}, time_point{std::chrono::nanoseconds(end_time)}));
    }
#endif
}

std::pair<uint32_t, uint32_t> PerformanceManager::gl_timer::assert_query(uint32_t index) {
    if (index > query_ids.size()) {
        throw std::runtime_error(
            ("gl_timer: non-coherent query IDs for timer " + conf.name + ", something is probably wrong.").c_str());
    }
    if (index == query_ids.size()) {
        std::array<uint32_t, 2> ids = {0, 0};
#ifdef MEGAMOL_USE_OPENGL
        glGenQueries(2, ids.data());
#endif
        query_ids.emplace_back(std::make_pair(ids[0], ids[1]));
    }
    return query_ids[index];
}

PerformanceManager::handle_vector PerformanceManager::add_timers(
    megamol::core::Module* m, std::vector<basic_timer_config> timer_list) {
    handle_vector ret;
    timer_config conf;
    conf.parent_pointer = m;
    conf.parent_type = parent_type::MODULE;
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

PerformanceManager::handle_vector PerformanceManager::add_timers(megamol::core::Call* c, query_api api) {
    handle_vector ret;
    const auto caps = c->GetCapabilities();
    timer_config conf;
    conf.parent_pointer = c;
    conf.parent_type = parent_type::CALL;
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

void PerformanceManager::remove_timers(handle_vector handles) {
    for (auto handle : handles) {
        timers.erase(handle);
    }
    handle_holes.insert(handle_holes.end(), handles.begin(), handles.end());
}

std::string PerformanceManager::parent_name(const timer_config& conf) {
    switch (conf.parent_type) {
    case parent_type::CALL: {
        const auto c = static_cast<megamol::core::Call*>(conf.parent_pointer);
        return c->GetDescriptiveText();
    }
    case parent_type::MODULE: {
        const auto m = static_cast<megamol::core::Module*>(conf.parent_pointer);
        return m->Name().PeekBuffer();
    }
    default:
        return "";
    }
}

std::string PerformanceManager::lookup_parent(handle_type h) {
    const auto& conf = timers[h]->get_conf();
    return parent_name(conf);
}

void* PerformanceManager::lookup_parent_pointer(handle_type h) {
    return timers[h]->get_conf().parent_pointer;
}

PerformanceManager::parent_type PerformanceManager::lookup_parent_type(handle_type h) {
    return timers[h]->get_conf().parent_type;
}

std::string PerformanceManager::lookup_name(handle_type h) {
    return timers[h]->get_conf().name;
}

const PerformanceManager::timer_config PerformanceManager::lookup_config(handle_type h) {
    return timers[h]->get_conf();
}

PerformanceManager::handle_vector PerformanceManager::lookup_timers(void* parent) {
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

PerformanceManager::handle_type PerformanceManager::add_timer(std::unique_ptr<Itimer> t) {
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

void PerformanceManager::endFrame() {
#ifdef MEGAMOL_USE_OPENGL
    int done = (gl_timer::last_query == 0);
    while (!done) {
        glGetQueryObjectiv(gl_timer::last_query, GL_QUERY_RESULT_AVAILABLE, &done);
    }
#endif

    frame_info this_frame;
    this_frame.frame = current_frame;

    for (auto& [key, timer] : timers) {
        if (timer->get_start_frame() != this_frame.frame) {
            // timer did not start this frame
            continue;
        } else {
            if (timer->started) {
                // timer was not ended this frame, that is not nice
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "PerformanceManager: timer %s was not properly ended in frame %u", timer->get_conf().name.c_str(),
                    this_frame.frame);
                continue;
            }
        }
        timer->collect();
        auto& tconf = timer->get_conf();
        timer_entry e;
        e.handle = timer->get_handle();
        e.user_index = tconf.user_index;

        for (uint32_t region = 0; region < timer->get_region_count(); ++region) {
            e.frame_index = region;
            e.api = tconf.api;

            e.start = timer->get_start(region);
            e.end = timer->get_end(region);
            e.duration = time_point{timer->get_end(region) - timer->get_start(region)};
            this_frame.entries.push_back(e);
        }
        std::sort(this_frame.entries.begin(), this_frame.entries.end(),
            [](timer_entry& a, timer_entry& b) { return a.start < b.start; });
    }

    for (auto& subscriber : subscribers) {
        subscriber(this_frame);
    }
}
} // namespace megamol::frontend_resources
