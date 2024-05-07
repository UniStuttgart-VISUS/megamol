#include "Timers.h"

#include <array>

#include "mmcore/Call.h"
#include "mmcore/Module.h"

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

timer_region& Itimer::start(frame_type frame) {
    if (frame != start_frame) {
        frame_index = 0;
    }
    timer_region r{zero_time, zero_time, current_global_index++, frame_index++, frame, {nullptr, nullptr}, false};

    start_frame = frame;
    return regions.emplace_back(r);
}

} // namespace megamol::frontend_resources::performance
