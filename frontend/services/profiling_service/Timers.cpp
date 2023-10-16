#include "Timers.h"

#include "mmcore/Call.h"
#include "mmcore/Module.h"

#include <array>

#ifdef MEGAMOL_USE_OPENGL
#include <glad/gl.h>
#endif

#define TIMERS_USE_DOUBLE_BUFFERING

namespace megamol::frontend_resources::performance {

std::string Itimer::parent_name(const timer_config& conf) {
    switch (conf.parent) {
    case parent_type::CALL: {
        const auto c = static_cast<megamol::core::Call*>(conf.parent_pointer);
        return c->GetDescriptiveText();
    }
    case parent_type::USER_REGION: {
        const auto m = static_cast<megamol::core::Module*>(conf.parent_pointer);
        return m->Name().PeekBuffer();
    }
    case parent_type::BUILTIN:
        return "BuiltIn";
    default:
        return "";
    }
}

bool Itimer::start(frame_type frame) {
    auto new_frame = false;
    if (frame != start_frame) {
        new_frame = true;
        //regions.clear();
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

void Itimer::end() {
    if (!started) {
        throw std::runtime_error(
            ("cpu_timer: region " + parent_name(conf) + "::" + conf.name + " needs to be started before being ended")
                .c_str());
    }
    started = false;
}

bool cpu_timer::start(frame_type frame) {
    const auto ret = Itimer::start(frame);
    //printf("starting region %s as %u\n", parent_name(this->conf).c_str(), current_global_index);
    timer_region r{time_point::clock::now(), time_point(), current_global_index++};
    regions.emplace_back(r);
    return ret;
}

void cpu_timer::end() {
    Itimer::end();
    regions.back().end = time_point::clock::now();
}

void cpu_timer::clear(frame_type frame) {
    this->regions.clear();
}

gl_timer::~gl_timer() {
#ifdef MEGAMOL_USE_OPENGL
    for (auto i = 0; i < frame_data.size(); ++i) {
        for (auto& q_pair : frame_data[i].query_ids) {
            glDeleteQueries(1, &q_pair.first);
            glDeleteQueries(1, &q_pair.second);
        }
    }
#endif
}

bool gl_timer::start(frame_type frame) {
    const auto new_frame = Itimer::start(frame);
    frame_chooser = choose_launch_buffer(frame);
    if (new_frame) {
        frame_data[frame_chooser].query_index = 0;
        frame_data[frame_chooser].frame = frame;
    }
    last_query[frame_chooser] = assert_query(frame_data[frame_chooser].query_index).first;
    frame_data[frame_chooser].frame_indices[frame_data[frame_chooser].query_index] = current_global_index++;
#ifdef MEGAMOL_USE_OPENGL
    glQueryCounter(last_query[frame_chooser], GL_TIMESTAMP);
#endif
    return new_frame;
}

void gl_timer::end() {
    Itimer::end();
    last_query[frame_chooser] = assert_query(frame_data[frame_chooser].query_index).second;
#ifdef MEGAMOL_USE_OPENGL
    glQueryCounter(last_query[frame_chooser], GL_TIMESTAMP);
#endif
    frame_data[frame_chooser].query_index++;
}

void gl_timer::wait_for_frame_end(frame_type frame) {
    auto lq = get_last_query(choose_collect_buffer(frame));
    //printf("blocking for frame %u (buffer %u)\n", frame, choose_collect_buffer(frame));
    int done = (lq == 0);
    //if (done)
    //    printf("actually, not blocking\n");
#ifdef MEGAMOL_USE_OPENGL
    while (!done) {
        glGetQueryObjectiv(lq, GL_QUERY_RESULT_AVAILABLE, &done);
    }
#endif
}

int32_t gl_timer::choose_launch_buffer(frame_type frame) {
#ifdef TIMERS_USE_DOUBLE_BUFFERING
    // the "one" frame
    return frame % 2;
#else
    // this should be identical to the basic implementation
    return 0;
#endif
}

int32_t gl_timer::choose_collect_buffer(frame_type frame) {
#ifdef TIMERS_USE_DOUBLE_BUFFERING
    // the "other" frame (see above)
    return (frame + 1) % 2;
#else
    return 0;
#endif
}

void gl_timer::collect(frame_type frame) {
#ifdef MEGAMOL_USE_OPENGL
    GLuint64 start_time, end_time;
    // we are already informed which frame we need to *really* collect
    const auto frame_to_collect = choose_launch_buffer(frame);
    //printf("collecting frame %u (buffer %u)\n", frame, frame_to_collect);
    for (uint32_t index = 0; index < frame_data[frame_to_collect].query_index; ++index) {
        const auto& [start, end] = frame_data[frame_to_collect].query_ids[index];
        glGetQueryObjectui64v(start, GL_QUERY_RESULT, &start_time);
        glGetQueryObjectui64v(end, GL_QUERY_RESULT, &end_time);
        timer_region r{time_point{std::chrono::nanoseconds(start_time)}, time_point{std::chrono::nanoseconds(end_time)},
            frame_data[frame_to_collect].frame_indices[index]};
        //printf("got %lld - %lld\n", r.start.time_since_epoch().count(), r.end.time_since_epoch().count());
        regions.emplace_back(r);
    }
#endif
}

void gl_timer::clear(frame_type frame) {
    this->regions.clear();
    frame_data[choose_launch_buffer(frame)].query_index = 0;
}

std::pair<uint32_t, uint32_t> gl_timer::assert_query(uint32_t index) {
    if (index > frame_data[frame_chooser].query_ids.size()) {
        throw std::runtime_error(
            ("gl_timer: non-coherent query IDs for timer " + conf.name + ", something is probably wrong.").c_str());
    }
    if (index == frame_data[frame_chooser].query_ids.size()) {
        std::array<uint32_t, 2> ids = {0, 0};
#ifdef MEGAMOL_USE_OPENGL
        glGenQueries(2, ids.data());
#endif
        frame_data[frame_chooser].query_ids.emplace_back(ids[0], ids[1]);
        frame_data[frame_chooser].frame_indices.resize(frame_data[frame_chooser].query_ids.size());
    }
    return frame_data[frame_chooser].query_ids[index];
}

void gl_timer::new_frame(frame_type frame) {
    last_query[choose_launch_buffer(frame)] = 0;
}

uint32_t gl_timer::get_last_query(int32_t chosen_frame_buffer) {
    return last_query[chosen_frame_buffer];
}

} // namespace megamol::frontend_resources::performance
