/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

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

namespace megamol::frontend_resources {

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

    using handle_type = uint32_t;
    using timer_index = uint32_t;
    using handle_vector = std::vector<handle_type>;
    using frame_type = uint32_t;
    using user_index_type = uint32_t;
    using time_point = std::chrono::steady_clock::time_point;

    enum class query_api { CPU, OPENGL }; // TODO: CUDA, OpenCL, Vulkan, whatnot

    static constexpr const char* query_api_string(query_api api) {
        switch (api) {
        case query_api::CPU:
            return "CPU";
        case query_api::OPENGL:
            return "OpenGL";
        }
        return "unknown";
    }

    enum class parent_type { CALL, USER_REGION, BUILTIN };

    static constexpr const char* parent_type_string(parent_type parent) {
        switch (parent) {
        case parent_type::CALL:
            return "Call";
        case parent_type::USER_REGION:
            return "UserRegion";
        case parent_type::BUILTIN:
            return "BuiltIn";
        }
        return "unknown";
    }

    struct basic_timer_config {
        std::string name = "unnamed";
        query_api api = query_api::CPU;
        user_index_type user_index = 0;
    };

    struct timer_config : public basic_timer_config {
        PerformanceManager::parent_type parent_type = parent_type::CALL;
        void* parent_pointer = nullptr;
        std::string comment;
    };

    struct timer_entry {
        // the user cannot fiddle with timers directly, this class needs to be asked
        handle_type handle = 0;
        query_api api = query_api::CPU;
        // local index inside one frame (if this region is touched multiple times per frame)
        uint32_t frame_index = 0;
        // user payload, used to track call indices, for example
        user_index_type user_index = 0;
        PerformanceManager::parent_type parent_type = parent_type::BUILTIN;
        time_point start, end, duration;
        int64_t global_index;
    };

    struct timer_region {
        time_point start, end;
        int64_t global_index = -1;
    };

    struct frame_info {
        frame_type frame = 0;
        std::vector<timer_entry> entries;
    };
    using update_callback = std::function<void(const frame_info&)>;

    class Itimer {
        friend class PerformanceManager;

    public:
        Itimer(timer_config conf) : conf(std::move(conf)) {}
        virtual ~Itimer() = default;

        [[nodiscard]] const timer_config& get_conf() const {
            return conf;
        }
        [[nodiscard]] handle_type get_handle() const {
            return h;
        }
        [[nodiscard]] uint32_t get_region_count() const {
            return regions.size();
        }
        [[nodiscard]] time_point get_start(uint32_t index) const {
            return regions[index].start;
        }
        [[nodiscard]] time_point get_end(uint32_t index) const {
            return regions[index].end;
        }
        [[nodiscard]] int64_t get_global_index(uint32_t index) const {
            return regions[index].global_index;
        }
        [[nodiscard]] frame_type get_start_frame() const {
            return start_frame;
        }

        void set_comment(std::string comment) {
            conf.comment = comment;
        }

    protected:
        // returns whether this is a new frame from what has been seen
        virtual bool start(frame_type frame);

        virtual void end();

        virtual void collect() = 0;

        timer_config conf;
        //time_point last_start;
        std::vector<timer_region> regions;
        bool started = false;
        frame_type start_frame = std::numeric_limits<frame_type>::max();
        handle_type h = 0;
    };

    class cpu_timer : public Itimer {
    public:
        cpu_timer(const timer_config& conf) : Itimer(conf) {}

        bool start(frame_type frame) override;

        void end() override;

    protected:
        void collect() override {}
    };

    class gl_timer : public Itimer {
        friend class PerformanceManager;

    public:
        gl_timer(const timer_config& conf) : Itimer(conf) {}

        ~gl_timer() override;

        bool start(frame_type frame) override;

        void end() override;

    protected:
        void collect() override;

    private:
        std::pair<uint32_t, uint32_t> assert_query(uint32_t index);

        std::vector<std::pair<uint32_t, uint32_t>> query_ids;
        std::vector<int32_t> frame_indices;
        uint32_t query_index = 0;
        inline static uint32_t last_query = 0;
    };

    // names and API defined explicitly for modules
    handle_vector add_timers(megamol::core::Module* m, std::vector<basic_timer_config> timer_list);

    // names derived from callbacks
    handle_vector add_timers(megamol::core::Call* c, query_api api);

    // explicit name
    handle_type add_timer(std::string name, query_api api);

    void remove_timers(handle_vector handles);

    // hint: this is not for free, so don't call this all the time
    static std::string parent_name(const timer_config& conf);

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

    void endFrame();

    handle_type current_handle = 0;
    std::vector<handle_type> handle_holes;
    std::unordered_map<handle_type, std::unique_ptr<Itimer>> timers;
    frame_type current_frame = 0;
    // there can only be one PerformanceManager currently.
    inline static int64_t current_global_index = 0;
    std::vector<update_callback> subscribers;

#ifdef MEGAMOL_USE_OPENGL
    handle_type whole_frame_gl;
#endif
    handle_type whole_frame_cpu;
};

} // namespace megamol::frontend_resources
