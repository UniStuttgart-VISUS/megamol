#pragma once

#include <chrono>
#include <exception>
#include "glad/glad.h"
#include "mmcore/Call.h"
#include "mmcore/Module.h"
#include "mmcore/utility/log/Log.h"

namespace megamol::frontend {
class Profiling_Service;
} // namespace megamol::frontend

namespace megamol::frontend_resources {

static std::string PerformanceManager_Req_Name = "PerformanceManager";

class PerformanceManager {
public:
    PerformanceManager();
    ~PerformanceManager();

    using handle_type = uint32_t;
    using timer_index = uint32_t;
    using handle_vector = std::vector<handle_type>;
    using frame_type = uint32_t;
    using time_point = std::chrono::steady_clock::time_point;

    enum class query_api { CPU, OPENGL }; // TODO: CUDA, OpenCL, Vulkan, whatnot

    enum class entry_type { START, END, DURATION };

    enum class parent_type { CALL, MODULE };

    struct basic_timer_config {
        std::string name = "unnamed";
        query_api api = query_api::CPU;
    };

    struct timer_config : public basic_timer_config {
        parent_type parent_type = parent_type::CALL;
        void* parent_pointer = nullptr;
    };

    struct timer_entry {
        handle_type handle = 0;
        entry_type type = entry_type::START;
        frame_type frame = 0;
        // local index inside one frame (if this region is touched multiple times per frame)
        uint32_t frame_index = 0;
        time_point timestamp;
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
        [[nodiscard]] handle_type get_handle() const {
            return h;
        }
        [[nodiscard]] uint32_t get_region_count() const {
            return regions.size();
        }
        [[nodiscard]] time_point get_start(uint32_t index) const {
            return regions[index].first;
        }
        [[nodiscard]] time_point get_end(uint32_t index) const {
            return regions[index].second;
        }
        [[nodiscard]] frame_type get_start_frame() const {
            return start_frame;
        }

    protected:
        // returns whether this is a new frame from what has been seen
        virtual bool start(frame_type frame) {
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

        virtual void end() {
            if (!started) {
                throw std::exception(("cpu_timer: region " + _conf.name + "needs to be started before being ended").c_str());
            }
            started = false;
        };
        virtual void collect() = 0;

        timer_config _conf;
        time_point last_start;
        std::vector<std::pair<time_point, time_point>> regions;
        bool started = false;
        frame_type start_frame = std::numeric_limits<frame_type>::max();
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

        bool start(frame_type frame) override {
            const auto ret = Itimer::start(frame);
            last_start = std::chrono::high_resolution_clock::now();
            return ret;
        }
        void end() override {
            Itimer::end();
            auto end = std::chrono::high_resolution_clock::now();
            regions.emplace_back(std::make_pair(last_start, end));
        }

    protected:
        void collect() override {}
    };

    class gl_timer : public Itimer {
        friend class PerformanceManager;

    public:
        gl_timer(const timer_config& conf) : Itimer(conf) {
        }

        ~gl_timer() override {
            for (auto& q_pair : query_ids) {
                glDeleteQueries(1, &q_pair.first);
                glDeleteQueries(1, &q_pair.second);
            }
        }

        bool start(frame_type frame) override {
            const auto new_frame = Itimer::start(frame);
            if (new_frame) {
                query_index = 0;
            }
            last_query = assert_query(query_index).first;
            glQueryCounter(last_query, GL_TIMESTAMP);
            return new_frame;
        }

        void end() override {
            Itimer::end();
            last_query = assert_query(query_index).second;
            glQueryCounter(last_query, GL_TIMESTAMP);
            query_index++;
        }

    protected:
        void collect() override {
            GLuint64 start_time, end_time;
            for (uint32_t index = 0; index < query_index; ++index) {
                const auto& [start, end] = query_ids[index];
                glGetQueryObjectui64v(start, GL_QUERY_RESULT, &start_time);
                glGetQueryObjectui64v(end, GL_QUERY_RESULT, &end_time);
                regions.emplace_back(std::make_pair(
                    time_point{std::chrono::nanoseconds(start_time)}, time_point{std::chrono::nanoseconds(end_time)}));
            }
        }

    private:
        std::pair<uint32_t, uint32_t> assert_query(uint32_t index) {
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

        std::vector<std::pair<uint32_t, uint32_t>> query_ids;
        uint32_t query_index = 0;
        inline static uint32_t last_query = 0;
    };

    // names and API defined explicitly for modules
    handle_vector add_timers(megamol::core::Module* m, std::vector<basic_timer_config> timer_list) {
        handle_vector ret;
        timer_config conf;
        conf.parent_pointer = m;
        conf.parent_type = parent_type::MODULE;
        for (const auto& btc : timer_list) {
            conf.api = btc.api;
            conf.name = btc.name;
            switch (conf.api) {
            case query_api::CPU: {
                cpu_timer t(conf);
                ret.push_back(add_timer(t));
                break;
            }
            case query_api::OPENGL: {
                gl_timer t(conf);
                ret.push_back(add_timer(t));
                break;
            }
            }
        }
        return ret;
    }

    // names and API derived from capabilities and callbacks
    // note: a CPU timer is added alongside all accelerator timers!
    handle_vector add_timers(megamol::core::Call* c) {
        handle_vector ret;
        const auto caps = c->GetCapabilities();
        timer_config conf;
        conf.parent_pointer = c;
        conf.parent_type = parent_type::CALL;
        for (auto i = 0; i < c->GetCallbackCount(); ++i) {
            if (caps.OpenGLRequired()) {
                conf.name = c->GetCallbackName(i) + "(GL)";
                conf.api = query_api::OPENGL;
                gl_timer t(conf);
                ret.push_back(add_timer(t));
            }
            conf.name = c->GetCallbackName(i);
            conf.api = query_api::CPU;
            cpu_timer t(conf);
            // TODO does this spawn copies? constructor fun? need for std::move??
            ret.push_back(add_timer(t));
        }
        return ret;
    }

    void remove_timers(handle_vector handles) {
        for (auto handle : handles) {
            timers.erase(handle);
        }
        handle_holes.insert(handle_holes.end(), handles.begin(), handles.end());
    }

    // hint: this is not for free, so don't call this all the time
    std::string lookup_parent(handle_type h) {
        const auto& conf = timers[h]->get_conf();
        auto p = conf.parent_pointer;
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

    // hint: this is not for free, so don't call this all the time
    std::string lookup_name(handle_type h) {
        return timers[h]->get_conf().name;
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
        handle_type my_handle = 0;
        if (!handle_holes.empty()) {
            my_handle = handle_holes.back();
            handle_holes.pop_back();
        } else {
            my_handle = current_handle;
            current_handle++;
        }
        t->h = my_handle;
        timers.emplace(my_handle, t);
        return my_handle;
    }

    void startFrame() {
        gl_timer::last_query = 0;
    }

    void endFrame() {
        int done = 0;
        do {
            glGetQueryObjectiv(gl_timer::last_query, GL_QUERY_RESULT_AVAILABLE, &done);
        } while (!done);

        frame_info this_frame;
        this_frame.frame = current_frame;

        for (auto& [key, timer] : timers) {
            if (timer->get_start_frame() != this_frame.frame) {
                // timer did not start this frame
                continue;
            } else {
                if (timer->started) {
                    // timer was not ended this frame, that is not nice
                    Log::DefaultLog.WriteWarn("PerformanceManager: timer %s was not properly ended in frame %u",
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

    handle_type current_handle = 0;
    //std::vector<timer> _timers;
    //std::unordered_map<handle_type, std::vector<timer>::size_type> timer_map;
    std::vector<handle_type> handle_holes;
    std::unordered_map<handle_type, timer> timers;
    frame_type current_frame = 0;
    std::vector<update_callback> subscribers;
};

} // namespace megamol::frontend_resources
