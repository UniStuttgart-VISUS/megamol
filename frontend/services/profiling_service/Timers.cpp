#include "Timers.h"

#include "mmcore/Call.h"
#include "mmcore/Module.h"

#include <algorithm>
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
    timer_region r{time_point::clock::now(), time_point(), current_global_index++, frame, {nullptr, nullptr}, false};
    regions.emplace_back(r);
    return ret;
}

void cpu_timer::end() {
    Itimer::end();
    regions.back().end = time_point::clock::now();
    regions.back().ready = true;
}

void cpu_timer::clear(frame_type frame) {
    this->regions.clear();
}

gl_timer::~gl_timer() {}

bool gl_timer::start(frame_type frame) {
    const auto new_frame = Itimer::start(frame);


    if (first_frame_) {
        timer_region r{time_point(), time_point(), current_global_index++, frame, {nullptr, nullptr}, false};
        r.qids[0] = std::make_shared<GLQuery>();
        r.qids[0]->Counter();
        regions.emplace_back(r);
        first_frame_ = false;
    }
    //#ifdef MEGAMOL_USE_OPENGL
    //    glGenQueries(2, r.qids.data());
    //    glQueryCounter(r.qids[0], GL_TIMESTAMP);
    //#endif

    return new_frame;
}

void gl_timer::end() {
    Itimer::end();

    regions.back().qids[1] = std::make_shared<GLQuery>();
    regions.back().qids[1]->Counter();

    timer_region r{time_point(), time_point(), current_global_index++, regions.back().frame + 1, {nullptr, nullptr}, false};
    r.qids[0] = regions.back().qids[1];
    regions.emplace_back(r);

    //#ifdef MEGAMOL_USE_OPENGL
    //    glQueryCounter(regions.back().qids[1], GL_TIMESTAMP);
    //#endif
}

//int32_t gl_timer::choose_launch_buffer(frame_type frame) {
//#ifdef TIMERS_USE_DOUBLE_BUFFERING
//    // the "one" frame
//    return frame % 2;
//#else
//    // this should be identical to the basic implementation
//    return 0;
//#endif
//}
//
//int32_t gl_timer::choose_collect_buffer(frame_type frame) {
//#ifdef TIMERS_USE_DOUBLE_BUFFERING
//    // the "other" frame (see above)
//    return (frame + 1) % 2;
//#else
//    return 0;
//#endif
//}

void gl_timer::collect(frame_type frame) {
#ifdef MEGAMOL_USE_OPENGL
    for (auto& r : regions) {
        GLuint64 start_time = 0, end_time = 0;
        bool start_ready = false, end_ready = false;
        if (r.start == time_point()) {
            if (r.qids[0])
                start_time = r.qids[0]->GetNW();
            //glGetQueryObjectui64v(r.qids[0], GL_QUERY_RESULT_NO_WAIT, &start_time);

            if (start_time) {
                r.start = time_point{std::chrono::nanoseconds(start_time)};
                /*glDeleteQueries(1, &r.qids[0]);
                r.qids[0] = 0;*/
                start_ready = true;
            }
        } else {
            start_ready = true;
        }
        if (r.end == time_point()) {
            if (r.qids[1])
                end_time = r.qids[1]->GetNW();
            //glGetQueryObjectui64v(r.qids[1], GL_QUERY_RESULT_NO_WAIT, &end_time);

            if (end_time) {
                r.end = time_point{std::chrono::nanoseconds(end_time)};
                /*glDeleteQueries(1, &r.qids[1]);
                r.qids[1] = 0;*/
                end_ready = true;
            }
        } else {
            end_ready = true;
        }
        r.ready = start_ready && end_ready;
    }
#endif
}

void gl_timer::clear(frame_type frame) {
    regions.erase(std::remove_if(regions.begin(), regions.end(), [](auto const& r) { return r.ready; }), regions.end());
}

} // namespace megamol::frontend_resources::performance
