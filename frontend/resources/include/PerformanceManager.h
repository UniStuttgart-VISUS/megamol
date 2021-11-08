#pragma once

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "glad/glad.h"
#include <chrono>

namespace megamol {
namespace frontend {
    class Profiling_Service;
}
}

namespace megamol {
namespace frontend_resources {

class PerformanceManager {
public:
    PerformanceManager();
    ~PerformanceManager();

    using handle_type = uint32_t;
    using timer_index = uint32_t;
    using handle_vector = std::vector<handle_type>;
    using frame_type = uint32_t;

    enum class query_api {
        CPU,
        OPENGL,
        CUDA
    };

    struct timer_config {
        std::string name;
        query_api api = query_api::CPU;
    };

    class Itimer {
        friend class PerformanceManager;
    public:
        Itimer(const timer_config& conf) : _conf(conf) {}
        virtual ~Itimer() {};

        [[nodiscard]] const timer_config& get_conf() const {
            return _conf;
        }
        [[nodiscard]] const handle_type get_handle() const {
            return h;
        }
        [[nodiscard]] std::chrono::time_point<std::chrono::steady_clock> get_start() const {
            return _start;
        }
        [[nodiscard]] std::chrono::time_point<std::chrono::steady_clock> get_end() const {
            return _end;
        }
        [[nodiscard]] frame_type get_start_frame() const { return _start_frame; }

    protected:
        virtual void start(frame_type frame) = 0;
        virtual void end() = 0;
        virtual void collect() = 0;

        timer_config _conf;
        std::chrono::time_point<std::chrono::steady_clock> _start, _end;
        bool _started = false;
        frame_type _start_frame = std::numeric_limits<frame_type>::max();
        handle_type h;
    };

    class timer {
    public:
        template<typename concrete_timer>
        timer(concrete_timer&& t) : storage{std::forward<concrete_timer>(t)}, getter {
            [](std::any& storage) -> Itimer& { return std::any_cast<concrete_timer&>(storage); }
        } {}

        Itimer *operator->(){return &getter(storage);}

    private:
        std::any storage;
        Itimer& (*getter)(std::any&);
    };

    class cpu_timer: public Itimer {
    public:
        cpu_timer(const timer_config& conf) : Itimer(conf){}

        void start(frame_type frame) override {
            _start = std::chrono::high_resolution_clock::now();
            _started = true;
            _start_frame = frame;
        }
        void end() override {
            if (_started) {
                _end = std::chrono::high_resolution_clock::now();
                _started = false;
            } else {
                throw std::exception("cpu_timer: region needs to be started before being ended");
            }
        }
    protected:
        void collect() override {}
    };

    class gl_timer : public Itimer {
        friend class PerformanceManager;
    public:
        gl_timer(const timer_config& conf) : Itimer(conf) {
            glGenQueries(1, &_start_id);
            glGenQueries(1, &_end_id);
        }

        ~gl_timer() override {
            glDeleteQueries(1, &_start_id);
            glDeleteQueries(1, &_end_id);
        }

        void start(frame_type frame) override {
            glQueryCounter(_start_id, GL_TIMESTAMP);
            _started = true;
            _start_frame = frame;
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

    protected:
        void collect() override {
            GLuint64 time;
            glGetQueryObjectui64v(_start_id, GL_QUERY_RESULT, &time);
            //_start = std::chrono::steady_clock::time_point{std::chrono::duration_cast<std::chrono::steady_clock::time_point::duration>(std::chrono::nanoseconds(time))};
            _start = std::chrono::steady_clock::time_point{std::chrono::nanoseconds(time)};
            glGetQueryObjectui64v(_end_id, GL_QUERY_RESULT, &time);
            _end = std::chrono::steady_clock::time_point{std::chrono::nanoseconds(time)};
        }

    private:
        uint32_t _start_id = 0;
        uint32_t _end_id = 0;
        inline static uint32_t _last_query = 0;
    };

    struct timer_entry {
        handle_type handle = 0;
        bool is_start = true;
        frame_type frame = 0;
        std::chrono::steady_clock::time_point timestamp;
    };

    struct frame_info {
        frame_type frame = 0;
        std::vector<timer_entry> entries;
    };

    // names and API defined explicitly for modules
    handle_vector add_timers(megamol::core::Module *m, std::vector<timer_config> timers);

    // names and API derived from capabilities and callbacks
    handle_vector add_timers(megamol::core::Call *c) {
        handle_vector ret;
        const auto caps = c->GetCapabilities();
        for (auto i = 0; i < c->GetCallbackCount(); ++i) {
            timer_config conf;
            conf.name = c->GetCallbackName(i);
            if (caps.OpenGLRequired()) {
                conf.api = query_api::OPENGL;
                gl_timer t(conf);
                ret.push_back(add_timer(t));
            } else {
                conf.api = query_api::CPU;
                cpu_timer t(conf);
                // does this spawn copies? constructor fun? need for std::move??
                ret.push_back(add_timer(t));
            }

        }
        return ret;
    }

    handle_type add_timer(timer t) {
        const handle_type my_handle = current_handle;
        t->h = my_handle;
        _timers.push_back(t);
        timer_map[my_handle] = _timers.size() - 1;
        current_handle++;
        return my_handle;
    }

    std::string get_timer_parent(handle_type h);
    std::string get_timer_name(handle_type h);

    void start_timer(handle_type h);
    void stop_timer(handle_type h);

private:
    friend class frontend::Profiling_Service;

    void startFrame() {
        gl_timer::_last_query = 0;
    }

    void endFrame() {
        int done = 0;
        do {
            glGetQueryObjectiv(gl_timer::_last_query, GL_QUERY_RESULT_AVAILABLE, &done);
        } while (!done);

        frame_info this_frame;
        this_frame.frame = current_frame;

        for (auto& t: _timers) {
            if (t->get_start_frame() != this_frame.frame) {
                // timer did not run this frame
                continue;
            }
            timer_entry e;
            // TODO add all info
            t->collect();
            auto& tconf = t->get_conf();
            e.handle = t->get_handle();
            e.frame = this_frame.frame;
            // add start and end
            // TODO

            //switch (t->get_conf().api) {
            //case query_api::OPENGL:
            //    // falls through!
            //case query_api::CPU:
            //    // todo: grab other properties
            //    this_frame.entries.push_back(e);
            //    break;
            //case query_api::CUDA:
            //    break;
            //default: ;
            //}
        }
        current_frame++;
    }

    handle_type current_handle = 0;
    std::vector<timer> _timers;
    std::unordered_map<handle_type, std::vector<timer>::size_type> timer_map;
    frame_type current_frame = 0;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */
