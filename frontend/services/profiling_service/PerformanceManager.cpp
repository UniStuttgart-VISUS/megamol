#include "PerformanceManager.h"

#include <array>
#include "mmcore/Call.h"
#include "mmcore/Module.h"

#ifdef WITH_GL
#include "glad/glad.h"
#endif

namespace megamol {
namespace frontend_resources {

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
            throw std::exception(("timer: region " + _conf.name + "needs to be ended before being started").c_str());
        }
        return new_frame;
    }

    void PerformanceManager::Itimer::end() {
        if (!started) {
            throw std::exception(
                ("cpu_timer: region " + _conf.name + "needs to be started before being ended").c_str());
        }
        started = false;
    }

    bool PerformanceManager::cpu_timer::start(frame_type frame) {
        const auto ret = Itimer::start(frame);
        last_start = std::chrono::high_resolution_clock::now();
        return ret;
    }

    void PerformanceManager::cpu_timer::end() {
        Itimer::end();
        auto end = std::chrono::high_resolution_clock::now();
        regions.emplace_back(std::make_pair(last_start, end));
    }

    PerformanceManager::gl_timer::~gl_timer() {
        for (auto& q_pair : query_ids) {
            glDeleteQueries(1, &q_pair.first);
            glDeleteQueries(1, &q_pair.second);
        }
    }

    bool PerformanceManager::gl_timer::start(frame_type frame) {
        const auto new_frame = Itimer::start(frame);
        if (new_frame) {
            query_index = 0;
        }
        last_query = assert_query(query_index).first;
        glQueryCounter(last_query, GL_TIMESTAMP);
        return new_frame;
    }

    void PerformanceManager::gl_timer::end() {
        Itimer::end();
        last_query = assert_query(query_index).second;
        glQueryCounter(last_query, GL_TIMESTAMP);
        query_index++;
    }

    void PerformanceManager::gl_timer::collect() {
        GLuint64 start_time, end_time;
        for (uint32_t index = 0; index < query_index; ++index) {
            const auto& [start, end] = query_ids[index];
            glGetQueryObjectui64v(start, GL_QUERY_RESULT, &start_time);
            glGetQueryObjectui64v(end, GL_QUERY_RESULT, &end_time);
            regions.emplace_back(std::make_pair(
                time_point{std::chrono::nanoseconds(start_time)}, time_point{std::chrono::nanoseconds(end_time)}));
        }
    }

    std::pair<uint32_t, uint32_t> PerformanceManager::gl_timer::assert_query(uint32_t index) {
        if (index > query_ids.size()) {
            throw std::exception(
                ("gl_timer: non-coherent query IDs for timer " + _conf.name + ", something is probably wrong.")
                    .c_str());
        }
        if (index == query_ids.size()) {
            std::array<uint32_t, 2> ids = {0, 0};
            glGenQueries(2, ids.data());
            query_ids.emplace_back(std::make_pair(ids[0], ids[1]));
        }
        return query_ids[index];
    }

    PerformanceManager::handle_vector PerformanceManager::add_timers(megamol::core::Module* m, std::vector<basic_timer_config> timer_list) {
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

    PerformanceManager::handle_vector PerformanceManager::add_timers(megamol::core::Call* c) {
        handle_vector ret;
        const auto caps = c->GetCapabilities();
        timer_config conf;
        conf.parent_pointer = c;
        conf.parent_type = parent_type::CALL;
        for (auto i = 0; i < c->GetCallbackCount(); ++i) {
            if (caps.OpenGLRequired()) {
                conf.name = c->GetCallbackName(i) + "(GL)";
                conf.api = query_api::OPENGL;
                ret.push_back(add_timer(std::make_unique<gl_timer>(conf)));
            }
            conf.name = c->GetCallbackName(i);
            conf.api = query_api::CPU;
            // TODO does this spawn copies? constructor fun? need for std::move??
            ret.push_back(add_timer(std::make_unique<cpu_timer>(conf)));
        }
        return ret;
    }

    void PerformanceManager::remove_timers(handle_vector handles) {
        for (auto handle : handles) {
            timers.erase(handle);
        }
        handle_holes.insert(handle_holes.end(), handles.begin(), handles.end());
    }

    std::string PerformanceManager::lookup_parent(handle_type h) {
        const auto& conf = timers[h]->get_conf();
        const auto p = conf.parent_pointer;
        switch (conf.parent_type) {
        case parent_type::CALL: {
            const auto c = static_cast<megamol::core::Call*>(p);
            return c->GetDescriptiveText();
        }
        case parent_type::MODULE: {
            const auto m = static_cast<megamol::core::Module*>(p);
            return m->Name().PeekBuffer();
        }
        default:
            return "";
        }
    }
    std::string PerformanceManager::lookup_name(handle_type h) {
        return timers[h]->get_conf().name;
    }

    void PerformanceManager::subscribe_to_updates(update_callback& cb) {
        subscribers.push_back(cb);
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
        int done = (gl_timer::last_query == 0);
        while (!done) {
            glGetQueryObjectiv(gl_timer::last_query, GL_QUERY_RESULT_AVAILABLE, &done);
        }

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
                        "PerformanceManager: timer %s was not properly ended in frame %u",
                        timer->get_conf().name.c_str(), this_frame.frame);
                    continue;
                }
            }
            timer->collect();
            auto& tconf = timer->get_conf();
            timer_entry e;
            e.handle = timer->get_handle();
            e.frame = this_frame.frame;

            for (uint32_t region = 0; region < timer->get_region_count(); ++region) {
                e.frame_index = region;

                e.type = entry_type::START;
                e.timestamp = timer->get_start(region);
                this_frame.entries.push_back(e);

                e.type = entry_type::END;
                e.timestamp = timer->get_end(region);
                this_frame.entries.push_back(e);

                e.type = entry_type::DURATION;
                e.timestamp = time_point{timer->get_end(region) - timer->get_start(region)};
                this_frame.entries.push_back(e);
            }
        }

        for (auto& subscriber : subscribers) {
            subscriber(this_frame);
        }

        current_frame++;
    }
} // namespace frontend_resources
} // namespace megamol
