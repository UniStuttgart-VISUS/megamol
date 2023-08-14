#include "RTXInstrument.h"

#ifdef MEGAMOL_USE_POWER

#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <thread>

#include <mmcore/utility/log/Log.h>

#include <ParquetWriter.h>

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

void RTXInstrument::UpdateConfigs(std::filesystem::path const& config_folder, int points, int count,
    std::chrono::milliseconds range, std::chrono::milliseconds timeout) {
    if (std::filesystem::is_directory(config_folder)) {
        std::for_each(rtx_instr_.begin(), rtx_instr_.end(), [&](auto const& instr) {
            auto const& name = instr.first;
            auto const fullpath = config_folder / (name + ".rtxcfg");
            if (std::filesystem::exists(fullpath)) {
                sol_state_["points"] = points;
                sol_state_["count"] = count;
                sol_state_["range"] = range.count();
                sol_state_["timeout"] = timeout.count();
                sol_state_.script(fullpath.string());
                rtx_config_["name"] = sol_state_[name];
            }
        });
        config_range_ = range;
    }
}

void RTXInstrument::ApplyConfigs() {
    std::for_each(rtx_instr_.begin(), rtx_instr_.end(), [&](auto& instr) {
        auto const& name = instr.first;
        auto& i = instr.second;
        auto fit = rtx_config_.find(name);
        if (fit == rtx_config_.end()) {
            core::utility::log::Log::DefaultLog.WriteWarn("[RTXInstrument]: No config found for device {}", name);
        } else {
            i.synchronise_clock();
            i.reset(true, true);
            fit->second.apply(i);
            i.reference_position(oscilloscope_reference_point::left);
            i.trigger_position(oscilloscope_quantity(0, "ms"));
            i.operation_complete();
            core::utility::log::Log::DefaultLog.WriteInfo("[RTXInstrument]: Configured device {}", name);
        }
    });
}

void RTXInstrument::StartMeasurement(std::filesystem::path const& output_folder) {
    if (!std::filesystem::is_directory(output_folder)) {
        core::utility::log::Log::DefaultLog.WriteError("[RTXInstrument]: Path {} is not a directory", output_folder);
        return;
    }

    int global_device_counter = 0;
    std::for_each(rtx_instr_.begin(), rtx_instr_.end(), [&global_device_counter](auto& instr) {
        auto& i = instr.second;
        i.on_operation_complete_ex([&global_device_counter](visa_instrument&) { ++global_device_counter; });
        i.acquisition(oscilloscope_acquisition_state::single);
        i.operation_complete_async();
    });

    // hammer trigger
    auto t_func = [&]() {
        std::chrono::system_clock::time_point last_trigger;

        while (global_device_counter < rtx_instr_.size()) {
            if (waiting_on_trigger()) {
                last_trigger = trigger();
            } else {
                break;
            }
            std::this_thread::sleep_for(config_range_ * 1.1f);
        }

        std::for_each(rtx_instr_.begin(), rtx_instr_.end(), [&](auto& instr) {
            auto const& name = instr.first;
            auto& i = instr.second;

            auto fit = rtx_config_.find(name);
            if (fit == rtx_config_.end()) {
                core::utility::log::Log::DefaultLog.WriteError(
                    "[RTXInstrument]: Could not find config for device {}", name);
                return;
            }

            auto const& config = fit->second;
            auto const num_channels = config.channels();
            if (num_channels == 0) {
                core::utility::log::Log::DefaultLog.WriteError(
                    "[RTXInstrument]: No configured channels");
                return;
            }
            std::vector<oscilloscope_channel> channels(num_channels);
            config.channels(channels.data(), channels.size());

            auto const all_waveforms = i.data(oscilloscope_waveform_points::maximum);

            auto const num_segments = i.history_segments();

            std::vector<std::unordered_map<std::string, std::variant<samples_t, timeline_t>>> values_map(num_segments);

            for (std::decay_t<decltype(num_segments)> s_idx = 0; s_idx < num_segments; ++s_idx) {
                auto const fullpath = output_folder / (instr.first + "_s" + std::to_string(s_idx) + ".parquet");
                auto const& waveform = all_waveforms[s_idx * num_channels].waveform();
                auto const sample_times = generate_timestamps_ns(waveform);
                auto const abs_times =
                    offset_timeline(sample_times, std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                      std::chrono::duration<float>(waveform.segment_offset())));
                auto const ltrg_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::time_point_cast<std::chrono::system_clock::duration>(last_trigger) -
                    std::chrono::system_clock::from_time_t(0));
                auto const wall_times = offset_timeline(abs_times, ltrg_ns);

                values_map[s_idx]["sample_times"] = sample_times;
                values_map[s_idx]["abs_times"] = abs_times;
                values_map[s_idx]["wall_times"] = wall_times;

                for (unsigned int c_idx = 0; c_idx < num_channels; ++c_idx) {
                    auto const tpn = name + "_" + channels[c_idx].label().text();
                    values_map[s_idx][tpn] = transform_waveform(waveform);
                }

                ParquetWriter(fullpath, values_map[s_idx]);
            }

        });
    };

    auto t_thread = std::thread(t_func);
    t_thread.detach();
}

bool RTXInstrument::waiting_on_trigger() const {
    return std::any_of(rtx_instr_.begin(), rtx_instr_.end(), [](auto const& instr) {
        auto const& i = instr.second;
        auto res = i.query("STAT:OPER:COND?\n");
        *strchr(res.as<char>(), '\n') = 0;
        auto const val = std::atoi(res.as<char>());
        if (val & 8) {
            return true;
        }
        return false;
    });
}

std::chrono::system_clock::time_point RTXInstrument::trigger() {}

void RTXInstrument::write_data(std::filesystem::path const& output_folder, oscilloscope_sample const& data) const {}

RTXInstrument::timeline_t RTXInstrument::generate_timestamps_ns(
    visus::power_overwhelming::oscilloscope_waveform const& waveform) const {

    auto const t_begin = waveform.time_begin();
    auto const t_end = waveform.time_end();
    auto const t_dis = waveform.sample_distance();
    auto const t_off = waveform.segment_offset();
    auto const r_length = waveform.record_length();

    auto const t_b_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<float>(t_begin));
    auto const t_d_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<float>(t_dis));

    timeline_t ret(r_length, t_b_ns.count());

    auto const t_d_ns_c = t_d_ns.count();

    std::inclusive_scan(
        ret.begin(), ret.end(), ret.begin(), [&t_d_ns_c](auto const& lhs, auto const& rhs) { return lhs + t_d_ns_c; });

    return ret;
}

RTXInstrument::timeline_t RTXInstrument::offset_timeline(timeline_t const& timeline, std::chrono::nanoseconds offset) const {
    timeline_t ret(timeline.begin(), timeline.end());

    std::transform(ret.begin(), ret.end(), ret.begin(), [o = offset.count()](auto const& val) { return val + o; });

    return ret;
}

} // namespace megamol::frontend

#endif
