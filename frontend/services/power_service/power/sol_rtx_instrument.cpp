#include "sol_rtx_instrument.h"

#ifdef MEGAMOL_USE_POWER

#include <sol/sol.hpp>

#include <power_overwhelming/rtx_instrument.h>
#include <power_overwhelming/rtx_instrument_configuration.h>

namespace megamol::power {

using namespace visus::power_overwhelming;

void sol_rtx_instrument(sol::state& lua) {
    auto rtx_instrument_table = lua.new_usertype<rtx_instrument>("rtx_instrument",
        sol::constructors<rtx_instrument(),
            rtx_instrument(const char* path, const visa_instrument::timeout_type timeout),
            rtx_instrument(std::reference_wrapper<bool>, const char*, const visa_instrument::timeout_type)>());
#if 0
    rtx_instrument_table["acquisition"] =
        sol::overload(static_cast<rtx_instrument& (rtx_instrument::*)(_In_ const oscilloscope_acquisition&,
                          _In_ const bool, _In_ const bool)>(&rtx_instrument::acquisition),
            static_cast<oscilloscope_acquisition_state (rtx_instrument::*)() const>(&rtx_instrument::acquisition)/*,
            static_cast<const rtx_instrument& (rtx_instrument::*)(_In_ const oscilloscope_acquisition_state,
                _In_ const bool) const>(&rtx_instrument::acquisition)*/);
#endif

    rtx_instrument_table["acquisition"] =
        static_cast<rtx_instrument& (rtx_instrument::*)(const oscilloscope_acquisition&, const bool)>(
            &rtx_instrument::acquisition);

    rtx_instrument_table["channel"] =
        static_cast<rtx_instrument& (rtx_instrument::*)(const oscilloscope_channel&)>(&rtx_instrument::channel);

    rtx_instrument_table["reference_position"] =
        static_cast<rtx_instrument& (rtx_instrument::*)(const oscilloscope_reference_point)>(
            &rtx_instrument::reference_position);

    rtx_instrument_table["trigger_position"] = &rtx_instrument::trigger_position;

    rtx_instrument_table["trigger"] =
        static_cast<rtx_instrument& (rtx_instrument::*)(const oscilloscope_trigger&)>(&rtx_instrument::trigger);

    lua.set_function("find_resources", [](const char*, const char*) {
        std::vector<rtx_instrument> ret;

        auto devices = visa_instrument::find_resources("0x0AAD", "0x01D6");

        for (auto d = devices.as<char>(); (d != nullptr) && (*d != 0); d += strlen(d) + 1) {
            ret.emplace_back(d);
        }

        return ret;
    });
}


void sol_oscilloscope_single_acquisition(sol::state& lua) {
    auto acq_table = lua.new_usertype<oscilloscope_acquisition>(
        "oscilloscope_single_acquisition", sol::constructors<oscilloscope_acquisition()>());

    acq_table["count"] = static_cast<oscilloscope_acquisition& (oscilloscope_acquisition::*)(const unsigned int)>(
        &oscilloscope_acquisition::count);

    acq_table["points"] = static_cast<oscilloscope_acquisition& (oscilloscope_acquisition::*)(const unsigned int)>(
        &oscilloscope_acquisition::points);

    acq_table["segmented"] = static_cast<oscilloscope_acquisition& (oscilloscope_acquisition::*)(const bool)>(
        &oscilloscope_acquisition::segmented);
}


void sol_oscilloscope_reference_point(sol::state& lua) {
    lua.new_enum<oscilloscope_reference_point>("oscilloscope_reference_point",
        {{"left", oscilloscope_reference_point::left}, {"middle", oscilloscope_reference_point::middle},
            {"right", oscilloscope_reference_point::right}});
}


void sol_oscilloscope_channel(sol::state& lua) {
    auto channel_table = lua.new_usertype<oscilloscope_channel>(
        "oscilloscope_channel", sol::constructors<oscilloscope_channel(const std::uint32_t),
                                    oscilloscope_channel(const std::uint32_t, const oscilloscope_channel&)>());

    channel_table["attenuation"] =
        static_cast<oscilloscope_channel& (oscilloscope_channel::*)(const oscilloscope_quantity&)>(
            &oscilloscope_channel::attenuation);

    channel_table["label"] = static_cast<oscilloscope_channel& (oscilloscope_channel::*)(const oscilloscope_label&)>(
        &oscilloscope_channel::label);

    channel_table["state"] =
        static_cast<oscilloscope_channel& (oscilloscope_channel::*)(const bool)>(&oscilloscope_channel::state);

    channel_table["range"] = static_cast<oscilloscope_channel& (oscilloscope_channel::*)(const oscilloscope_quantity&)>(
        &oscilloscope_channel::range);

    channel_table["offset"] =
        static_cast<oscilloscope_channel& (oscilloscope_channel::*)(const oscilloscope_quantity&)>(
            &oscilloscope_channel::offset);
}


void sol_oscilloscope_edge_trigger(sol::state& lua) {
    auto trigger_table = lua.new_usertype<oscilloscope_trigger>(
        "oscilloscope_trigger", sol::constructors<oscilloscope_trigger(const char*, const char*)>());

    trigger_table["level"] = sol::overload(
        static_cast<oscilloscope_trigger& (oscilloscope_trigger::*)(const oscilloscope_trigger::input_type,
            const oscilloscope_quantity&)>(&oscilloscope_trigger::level),
        static_cast<oscilloscope_trigger& (oscilloscope_trigger::*)(const oscilloscope_quantity&)>(
            &oscilloscope_trigger::level));

    trigger_table["slope"] =
        static_cast<oscilloscope_trigger& (oscilloscope_trigger::*)(const oscilloscope_trigger_slope)>(
            &oscilloscope_trigger::slope);

    trigger_table["mode"] =
        static_cast<oscilloscope_trigger& (oscilloscope_trigger::*)(const oscilloscope_trigger_mode)>(
            &oscilloscope_trigger::mode);

    lua.new_enum<oscilloscope_trigger_slope>("oscilloscope_trigger_slope",
        {{"both", oscilloscope_trigger_slope::both}, {"rising", oscilloscope_trigger_slope::rising},
            {"falling", oscilloscope_trigger_slope::falling}});

    lua.new_enum<oscilloscope_trigger_mode>("oscilloscope_trigger_mode",
        {{"automatic", oscilloscope_trigger_mode::automatic}, {"normal", oscilloscope_trigger_mode::normal}});
}


void sol_oscilloscope_quantity(sol::state& lua) {
    auto quant_table = lua.new_usertype<oscilloscope_quantity>(
        "oscilloscope_quantity", sol::constructors<oscilloscope_quantity(const float, const char* unit)>());
}


void sol_oscilloscope_label(sol::state& lua) {
    auto label_table = lua.new_usertype<oscilloscope_label>(
        "oscilloscope_label", sol::constructors<oscilloscope_label(), oscilloscope_label(const char*, const bool)>());
}


void sol_rtx_instrument_configuration(sol::state& lua) {
    auto config_table = lua.new_usertype<rtx_instrument_configuration>(
        "rtx_instrument_configuration", sol::constructors<rtx_instrument_configuration(),
                                            rtx_instrument_configuration(const oscilloscope_quantity,
                                                std::reference_wrapper<const oscilloscope_acquisition>,
                                                std::reference_wrapper<const oscilloscope_trigger>, std::uint32_t)>());

    config_table["channel"] = &rtx_instrument_configuration::channel;

    //config_table["as_slave"] = &rtx_instrument_configuration::as_slave;
    lua.set_function("as_slave",
        [](const rtx_instrument_configuration& config,
            oscilloscope_quantity const& level) -> rtx_instrument_configuration { return config.as_slave(0, level); });

    //lua->set_function("get_config",
    //    [](const oscilloscope_quantity quant, const oscilloscope_acquisition& acq) -> rtx_instrument_configuration {
    //        /*oscilloscope_edge_trigger trigger = oscilloscope_edge_trigger("EXT");
    //        trigger.level(5, oscilloscope_quantity(2000.0f, "mV"))
    //            .slope(oscilloscope_trigger_slope::rising)
    //            .mode(oscilloscope_trigger_mode::normal);*/

    //        return rtx_instrument_configuration(quant, acq,
    //            dynamic_cast<oscilloscope_edge_trigger&>(oscilloscope_edge_trigger("EXT")
    //                .level(5, oscilloscope_quantity(2, "V"))
    //                .slope(oscilloscope_trigger_slope::rising)
    //                .mode(oscilloscope_trigger_mode::normal)),
    //            10000);
    //    });

    //lua->set_function(
    //    "get_trigger", [](const char* source) -> oscilloscope_edge_trigger {
    //        /*oscilloscope_edge_trigger trigger = oscilloscope_edge_trigger("EXT");
    //        trigger.level(5, oscilloscope_quantity(2000.0f, "mV"))
    //            .slope(oscilloscope_trigger_slope::rising)
    //            .mode(oscilloscope_trigger_mode::normal);*/

    //        return oscilloscope_edge_trigger(source);
    //    });
}

void sol_expressions(sol::state& lua,
    std::vector<std::unordered_map<std::string, std::variant<std::vector<float>, std::vector<int64_t>>>> const&
        val_map) {
    lua.set_function("rtx_plus", [&val_map](int idx, sol::variadic_args va) -> std::vector<float> {
        std::vector<float> ret;
        auto const& curr_map = val_map.at(idx);
        for (auto v : va) {
            std::string name = v;
            auto const& v_values = curr_map.at(name);
            if (!std::holds_alternative<std::vector<float>>(v_values)) {
                throw std::runtime_error("Unexpected type");
            }
            auto const& values = std::get<std::vector<float>>(v_values);
            if (ret.empty()) {
                ret = values;
            } else {
                std::transform(values.begin(), values.end(), ret.begin(), ret.begin(), std::plus<float>());
            }
        }
        return ret;
    });
    auto m_func_1 = [&val_map](int idx, sol::variadic_args va) -> std::vector<float> {
        std::vector<float> ret;
        auto const& curr_map = val_map.at(idx);
        for (auto v : va) {
            std::string name = v;
            auto const& v_values = curr_map.at(name);
            if (!std::holds_alternative<std::vector<float>>(v_values)) {
                throw std::runtime_error("Unexpected type");
            }
            auto const& values = std::get<std::vector<float>>(v_values);
            if (ret.empty()) {
                ret = values;
            } else {
                std::transform(values.begin(), values.end(), ret.begin(), ret.begin(), std::multiplies<float>());
            }
        }
        return ret;
    };
    auto m_func_2 = [&val_map](int idx, std::vector<float> const& lhs, std::string rhs) -> std::vector<float> {
        auto ret = lhs;
        auto const& v_values = val_map.at(idx).at(rhs);
        if (!std::holds_alternative<std::vector<float>>(v_values)) {
            throw std::runtime_error("Unexpected type");
        }
        auto const& values = std::get<std::vector<float>>(v_values);
        std::transform(values.begin(), values.end(), ret.begin(), ret.begin(), std::multiplies<float>());
        return ret;
    };
    auto m_func_3 = [&val_map](int idx, std::string lhs, std::vector<float> const& rhs) -> std::vector<float> {
        auto ret = rhs;
        auto const& v_values = val_map.at(idx).at(lhs);
        if (!std::holds_alternative<std::vector<float>>(v_values)) {
            throw std::runtime_error("Unexpected type");
        }
        auto const& values = std::get<std::vector<float>>(v_values);
        std::transform(values.begin(), values.end(), ret.begin(), ret.begin(), std::multiplies<float>());
        return ret;
    };

    lua.set_function("rtx_multiplies", sol::overload(m_func_1, m_func_2, m_func_3));
}


void sol_register_all(sol::state& lua) {
    sol_rtx_instrument(lua);

    sol_rtx_instrument_configuration(lua);

    sol_oscilloscope_single_acquisition(lua);

    sol_oscilloscope_reference_point(lua);

    sol_oscilloscope_channel(lua);

    sol_oscilloscope_edge_trigger(lua);

    sol_oscilloscope_quantity(lua);

    sol_oscilloscope_label(lua);
}

} // namespace megamol::power

#endif
