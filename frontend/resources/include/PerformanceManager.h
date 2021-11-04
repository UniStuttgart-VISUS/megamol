#pragma once

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "glad/glad.h"
#include <chrono>

namespace megamol {
namespace frontend_resources {

class PerformanceManager {
public:
    PerformanceManager();
    ~PerformanceManager();

    using handle_type = uint32_t;
    using handle_vector = std::vector<handle_type>;
    using frame_type = uint32_t;

    enum class query_api {
        CPU,
        OPENGL,
        CUDA
    };

    struct timer_config {
        std::string name;
        query_api api;
    };

    class timer {
    public:
        timer(timer_config conf) : _conf(conf) {}
        ~timer();

        const timer_config& get_conf() const { return _conf; }

        virtual void start() = 0;
        virtual void end() = 0;

    protected:
        std::chrono::time_point<std::chrono::steady_clock> get_start() {
            return _start;
        }
        std::chrono::time_point<std::chrono::steady_clock> get_end() {
            return _end;
        }

        timer_config _conf;
        std::chrono::time_point<std::chrono::steady_clock> _start, _end;
        bool _started = false;
    };

    class cpu_timer: timer {
    public:
        void start() override {
            _start = std::chrono::high_resolution_clock::now();
            _started = true;
        }
        void end() override {
            if (_started) {
                _end = std::chrono::high_resolution_clock::now();
                _started = false;
            } else {
                throw std::exception("cpu_timer: region needs to be started before being ended");
            }
        }
    };

    class gl_timer : timer {
    public:
        gl_timer(timer_config conf) : timer(conf) {
            glGenQueries(1, &_start_id);
            glGenQueries(1, &_end_id);
        }

        ~gl_timer() {
            glDeleteQueries(1, &_start_id);
            glDeleteQueries(1, &_end_id);
        }

        void start() override {
            glQueryCounter(_start_id, GL_TIMESTAMP);
            _started = true;
        }

        void end() override {
            if (_started) {
                glQueryCounter(_end_id, GL_TIMESTAMP);
                _started = false;
                _last_query = _end_id;
            } else {
                throw std::exception("gl_timer: region needs to be started before being ended");
            }
        }

    private:
        void collect() {
            GLuint64 time;
            glGetQueryObjectui64v(_start_id, GL_QUERY_RESULT, &time);
            //_start = std::chrono::steady_clock::time_point{std::chrono::duration_cast<std::chrono::steady_clock::time_point::duration>(std::chrono::nanoseconds(time))};
            _start = std::chrono::steady_clock::time_point{std::chrono::nanoseconds(time)};
            glGetQueryObjectui64v(_end_id, GL_QUERY_RESULT, &time);
            _end = std::chrono::steady_clock::time_point{std::chrono::nanoseconds(time)};
        }

        friend class PerformanceManager;
        uint32_t _start_id = 0;
        uint32_t _end_id = 0;
        inline static uint32_t _last_query = 0;
    };

    struct timer_entry {
        handle_type h;

    };

    struct frame_info {
        frame_type frame;
        std::vector<timer_entry> entries;
    };

    // names and API defined explicitly for modules
    handle_vector add_timers(megamol::core::Module *m, std::vector<timer_config> timers);
    // names and API derived from capabilities and callbacks
    handle_vector add_timers(megamol::core::Call *c);

    void start_timer(handle_type h);
    void stop_timer(handle_type h);

private:
    void startFrame(frame_type f) {
        gl_timer::_last_query = 0;
        current_frame = f;
    }

    void endFrame() {
        int done = 0;
        do {
            glGetQueryObjectiv(gl_timer::_last_query, GL_QUERY_RESULT_AVAILABLE, &done);
        } while (!done);

        frame_info this_frame;
        this_frame.frame = current_frame;

        for (auto& t: _timers) {
            // todo: can we check if it even ran this frame?
            timer_entry e;
            // add all info
            switch (t.get_conf().api) {
            case query_api::OPENGL:
                dynamic_cast<gl_timer&>(t).collect();
                // falls through!
            case query_api::CPU:
                // todo: grab other properties
                this_frame.entries.push_back(e);
                break;
            case query_api::CUDA:
                break;
            default: ;
            }
        }
    }

    handle_type current_handle = 0;
    std::vector<timer> _timers;
    frame_type current_frame = 0;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
