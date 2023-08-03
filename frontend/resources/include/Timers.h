/**
 * MegaMol
 * Copyright (c) 2023, MegaMol Dev Team
 * All rights reserved.
 */
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <vector>

namespace megamol::frontend_resources::performance {

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

struct timer_region {
    time_point start, end;
    int64_t global_index = -1;
};

struct basic_timer_config {
    std::string name = "unnamed";
    query_api api = query_api::CPU;
    user_index_type user_index = 0;
};

struct timer_config : public basic_timer_config {
    parent_type parent_type = parent_type::CALL;
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
    parent_type parent_type = parent_type::BUILTIN;
    time_point start, end, duration;
    int64_t global_index = 0;
};

struct frame_info {
    frame_type frame = 0;
    std::vector<timer_entry> entries;
};

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

    // hint: this is not for free, so don't call this all the time
    static std::string parent_name(const timer_config& conf);

    void set_comment(std::string comment) {
        conf.comment = comment;
    }

protected:
    // returns whether this is a new frame from what has been seen
    virtual bool start(frame_type frame);

    virtual void end();

    virtual void collect(frame_type frame) = 0;

    timer_config conf;
    //time_point last_start;
    std::vector<timer_region> regions;
    bool started = false;
    frame_type start_frame = std::numeric_limits<frame_type>::max();
    handle_type h = 0;
    // there can only be one PerformanceManager currently.
    inline static int64_t current_global_index = 0;
};

class cpu_timer : public Itimer {
public:
    cpu_timer(const timer_config& cfg) : Itimer(cfg) {}

    bool start(frame_type frame) override;

    void end() override;

protected:
    void collect(frame_type frame) override {}
};

class gl_timer : public Itimer {
    friend class PerformanceManager;

public:
    gl_timer(const timer_config& cfg) : Itimer(cfg) {}

    ~gl_timer() override;

    bool start(frame_type frame) override;

    void end() override;

    static void wait_for_frame_end(frame_type frame);

protected:
    void collect(frame_type frame) override;

private:
    static void new_frame(frame_type frame);
    static uint32_t get_last_query(int32_t chosen_frame_buffer);

    static int32_t choose_launch_buffer(frame_type frame);
    static int32_t choose_collect_buffer(frame_type frame);


    std::pair<uint32_t, uint32_t> assert_query(uint32_t index);

    // we need all of these for even and odd frames, I think
    struct frame_data_ {
        std::vector<std::pair<uint32_t, uint32_t>> query_ids;
        std::vector<int32_t> frame_indices;
        uint32_t query_index = 0;
        frame_type frame = 0;
        //inline static uint32_t last_query = 0;
    };
    std::array<frame_data_, 2> frame_data;
    inline static std::array<uint32_t, 2> last_query = {0, 0};
    int32_t frame_chooser = 0;
};

} // namespace megamol::frontend_resources::performance
