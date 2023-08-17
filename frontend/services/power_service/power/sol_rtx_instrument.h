#pragma once

#include <unordered_map>

#include <sol/sol.hpp>

namespace megamol::power {

void sol_rtx_instrument(sol::state& lua);

void sol_oscilloscope_single_acquisition(sol::state& lua);

void sol_oscilloscope_reference_point(sol::state& lua);

void sol_oscilloscope_channel(sol::state& lua);

void sol_oscilloscope_edge_trigger(sol::state& lua);

void sol_oscilloscope_quantity(sol::state& lua);

void sol_oscilloscope_label(sol::state& lua);

void sol_rtx_instrument_configuration(sol::state& lua);

void sol_expressions(sol::state& lua,
    std::vector<std::unordered_map<std::string, std::variant<std::vector<float>, std::vector<int64_t>>>> const&
        val_map);

void sol_register_all(sol::state& lua);

} // namespace visus::power_overwhelming
