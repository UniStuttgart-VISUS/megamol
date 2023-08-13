#include "RTXInstrument.h"

#ifdef MEGAMOL_USE_POWER

#include <algorithm>
#include <stdexcept>

#include <mmcore/utility/log/Log.h>

using namespace visus::power_overwhelming;

namespace megamol::frontend {

RTXInstrument::RTXInstrument() {
    auto num_devices = rtx_instrument::all(nullptr, 0);
    if (num_devices == 0)
        throw std::runtime_error("No RTX devices attached");
    std::vector<visus::power_overwhelming::rtx_instrument> rtx_instr(num_devices);
    rtx_instrument::all(rtx_instr.data(), rtx_instr.size());

    rtx_instr_.reserve(rtx_instr.size());

    std::for_each(rtx_instr.begin(), rtx_instr.end(), [&](rtx_instrument& i) {
        auto name = get_name(i);
        core::utility::log::Log::DefaultLog.WriteInfo("[RTXInstrument]: Found {} as {}", name, get_identity(i));
        rtx_instr_[get_name(i)] = std::move(i);
    });

    sol_state_.open_libraries(sol::lib::base);
}

void RTXInstrument::UpdateConfigs(std::filesystem::path const& config_folder) {
    if (std::filesystem::is_directory(config_folder)) {
        std::for_each(rtx_instr_.begin(), rtx_instr_.end(), [&](auto const& instr) {
            auto const name = instr.first;
            auto const fullpath = config_folder / (name + ".rtxcfg");
            if (std::filesystem::exists(fullpath)) {
                sol_state_.script(fullpath.string());
            }
        });
    }
}

} // namespace megamol::frontend

#endif
