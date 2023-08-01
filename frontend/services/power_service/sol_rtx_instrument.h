#pragma once

#include <memory>

#include <sol/sol.hpp>

namespace visus::power_overwhelming {

void sol_rtx_instrument(sol::state& lua);

void sol_oscilloscope_single_acquisition(sol::state& lua);

void sol_oscilloscope_reference_point(sol::state& lua);

void sol_oscilloscope_channel(sol::state& lua);

void sol_oscilloscope_edge_trigger(sol::state& lua);

void sol_oscilloscope_quantity(sol::state& lua);

void sol_oscilloscope_label(sol::state& lua);

void sol_rtx_instrument_configuration(sol::state& lua);

void sol_register_all(sol::state& lua);

} // namespace visus::power_overwhelming
