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
            ("timer: region " + parent_name(conf) + "::" + conf.name + " needs to be started before being ended")
                .c_str());
    }
    started = false;
}

} // namespace megamol::frontend_resources::performance
