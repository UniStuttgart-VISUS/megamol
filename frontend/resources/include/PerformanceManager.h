#pragma once

#include <chrono>
#include "glad/glad.h"
#include "mmcore/Call.h"
#include "mmcore/Module.h"

namespace megamol::frontend {
class Profiling_Service;
} // namespace megamol::frontend

namespace megamol::frontend_resources {

class PerformanceManager {
public:
    PerformanceManager();
    ~PerformanceManager();

    using handle_type = uint32_t;
    using timer_index = uint32_t;
    using handle_vector = std::vector<handle_type>;
    using frame_type = uint32_t;

    enum class query_api { CPU, OPENGL, CUDA };

    enum class entry_type { START, END, DURATION };

    enum class parent_type { CALL, MODULE };

    struct timer_config {
        parent_type parent_type = parent_type::CALL;
        void* parent_pointer = nullptr;
        std::string name;
        query_api api = query_api::CPU;
    };

    struct timer_entry {
        handle_type handle = 0;
        entry_type type = entry_type::START;
        frame_type frame = 0;
        std::chrono::steady_clock::time_point timestamp;
    };

    struct frame_info {
        frame_type frame = 0;
        std::vector<timer_entry> entries;
    };
    using update_callback = std::function<void(frame_info)>;

    class Itimer {
        friend class PerformanceManager;

    public:
        Itimer(const timer_config& conf) : _conf(conf) {}
        virtual ~Itimer(){};

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
        [[nodiscard]] frame_type get_start_frame() const {
            return _start_frame;
        }

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
        timer(concrete_timer&& t)
                : storage{std::forward<concrete_timer>(t)}
                , getter{[](std::any& a_storage) -> Itimer& { return std::any_cast<concrete_timer&>(a_storage); }} {}

        Itimer* operator->() {
            return &getter(storage);
        }

    private:
        std::any storage;
        Itimer& (*getter)(std::any&);
    };

    class cpu_timer : public Itimer {
    public:
        cpu_timer(const timer_config& conf) : Itimer(conf) {}

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
            //_start =
            // std::chrono::steady_clock::time_point{std::chrono::duration_cast<std::chrono::steady_clock::time_point::duration>(std::chrono::nanoseconds(time))};
            _start = std::chrono::steady_clock::time_point{std::chrono::nanoseconds(time)};
            glGetQueryObjectui64v(_end_id, GL_QUERY_RESULT, &time);
            _end = std::chrono::steady_clock::time_point{std::chrono::nanoseconds(time)};
        }

    private:
        uint32_t _start_id = 0;
        uint32_t _end_id = 0;
        inline static uint32_t _last_query = 0;
    };

    // names and API defined explicitly for modules
    handle_vector add_timers(megamol::core::Module* m, std::vector<timer_config> timers);

    // names and API derived from capabilities and callbacks
    handle_vector add_timers(megamol::core::Call* c) {
        handle_vector ret;
        const auto caps = c->GetCapabilities();
        for (auto i = 0; i < c->GetCallbackCount(); ++i) {
            timer_config conf;
            conf.name = c->GetCallbackName(i);
            // conf.parent = c->
            if (caps.OpenGLRequired()) {
                conf.api = query_api::OPENGL;
                gl_timer t(conf);
                ret.push_back(add_timer(t));
            } else {
                conf.api = query_api::CPU;
                cpu_timer t(conf);
                // TODO does this spawn copies? constructor fun? need for std::move??
                ret.push_back(add_timer(t));
            }
        }
        return ret;
    }

    std::string lookup_parent(handle_type h) {
        const auto& conf = _timers[timer_map[h]]->get_conf();
        auto p = conf.parent_pointer;
        switch (conf.parent_type) {
        case parent_type::CALL: {
            const auto c = static_cast<megamol::core::Call*>(p);
            // TODO calls need a proper tag "blah::blub->yada::oink"
            return c->ClassName();
        }
        case parent_type::MODULE: {
            const auto m = static_cast<megamol::core::Module*>(p);
            return m->Name().PeekBuffer();
        }
        default:
            return "";
        }
    }

    std::string lookup_name(handle_type h) {
        return _timers[timer_map[h]]->get_conf().name;
    }

    void subscribe_to_updates(update_callback& cb) {
        subscribers.push_back(cb);
    }

    std::string get_timer_parent(handle_type h);
    std::string get_timer_name(handle_type h);

    void start_timer(handle_type h);
    void stop_timer(handle_type h);

private:
    friend class frontend::Profiling_Service;

    handle_type add_timer(timer t) {
        const handle_type my_handle = current_handle;
        t->h = my_handle;
        _timers.push_back(t);
        timer_map[my_handle] = _timers.size() - 1;
        current_handle++;
        return my_handle;
    }

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

        for (auto& t : _timers) {
            if (t->get_start_frame() != this_frame.frame) {
                // timer did not run this frame
                continue;
            }
            t->collect();
            auto& tconf = t->get_conf();
            timer_entry e;
            e.handle = t->get_handle();
            e.frame = this_frame.frame;
            e.type = entry_type::START;
            e.timestamp = t->get_start();
            this_frame.entries.push_back(e);

            e.type = entry_type::END;
            e.timestamp = t->get_end();
            this_frame.entries.push_back(e);

            e.type = entry_type::DURATION;
            e.timestamp = std::chrono::time_point<std::chrono::steady_clock>{t->get_end() - t->get_start()};
            this_frame.entries.push_back(e);
        }
        // TODO can/need we move this? probably not needed anyway.
        frame_log.push_back(this_frame);

        for (auto& subscriber : subscribers) {
            // TODO this is crap: each subscriber must first find the relevant updates!
            subscriber(this_frame);
        }

        current_frame++;
    }

    handle_type current_handle = 0;
    std::vector<timer> _timers;
    std::unordered_map<handle_type, std::vector<timer>::size_type> timer_map;
    frame_type current_frame = 0;
    // TODO should probably not be here
    std::vector<frame_info> frame_log;
    std::vector<update_callback> subscribers;
};

} // namespace megamol::frontend_resources
