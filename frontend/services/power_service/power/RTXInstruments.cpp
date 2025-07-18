#include "RTXInstruments.h"

#ifdef MEGAMOL_USE_POWER

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <functional>
#include <future>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#include <mmcore/utility/log/Log.h>

#include "ColumnNames.h"
#include "Trigger.h"
#include "sol_rtx_instrument.h"

using namespace visus::power_overwhelming;

namespace megamol::power {

RTXInstruments::RTXInstruments(std::shared_ptr<Trigger> trigger) : trigger_(trigger) {
    auto num_devices = rtx_instrument::all(nullptr, 0);
    if (num_devices == 0)
        throw std::runtime_error("No RTX devices attached");
    std::vector<visus::power_overwhelming::rtx_instrument> rtx_instr(num_devices);
    rtx_instrument::all(rtx_instr.data(), rtx_instr.size());

    rtx_instr_.reserve(rtx_instr.size());

    std::for_each(rtx_instr.begin(), rtx_instr.end(), [&](rtx_instrument& i) {
        auto const name = get_pwrowg_str<visus::power_overwhelming::rtx_instrument>(
            i, &visus::power_overwhelming::rtx_instrument::name);
        core::utility::log::Log::DefaultLog.WriteInfo("[RTXInstruments]: Found %s as %s", name,
            get_pwrowg_str<visus::power_overwhelming::rtx_instrument>(
                i, &visus::power_overwhelming::rtx_instrument::identify));
        rtx_instr_[name] = std::move(i);
    });

    sol_state_.open_libraries(sol::lib::base);

    sol_register_all(sol_state_);
}

void RTXInstruments::UpdateConfigs(std::filesystem::path const& config_folder, int points, int count,
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
                sol_state_.script_file(fullpath.string());
                rtx_config_[name] = sol_state_[name];
            }
        });
        config_range_ = range;
    }
}

void RTXInstruments::ApplyConfigs(MetaData* meta) {
    try {
        if (meta) {
            meta->oszi_configs.clear();
        }
        std::for_each(rtx_instr_.begin(), rtx_instr_.end(), [&](auto& instr) {
            auto const& name = instr.first;
            auto& i = instr.second;
            auto fit = rtx_config_.find(name);
            if (fit == rtx_config_.end()) {
                core::utility::log::Log::DefaultLog.WriteWarn("[RTXInstruments]: No config found for device %s", name);
            } else {
                i.synchronise_clock(true);
                i.reset(rtx_instrument_reset::buffers | rtx_instrument_reset::status | rtx_instrument_reset::stop);
                // need to stop running measurement otherwise wait on trigger cannot be used to guard start of trigger sequence
                fit->second.beep_on_trigger(true).beep_on_apply(true).beep_on_error(true);
                if (!fit->second.slave()) {
                    main_instr_ = &i;
                }
                fit->second.apply(i);
                if (meta) {
                    auto const cfg_str_size = rtx_instrument_configuration::serialise(nullptr, 0, fit->second);
                    std::string cfg_str;
                    cfg_str.resize(cfg_str_size);
                    rtx_instrument_configuration::serialise(cfg_str.data(), cfg_str.size(), fit->second);
                    meta->oszi_configs[fit->first] = cfg_str;
                }
                i.reference_position(oscilloscope_reference_point::left);
                i.trigger_position(oscilloscope_quantity(0, "ms"));
                i.operation_complete();
                core::utility::log::Log::DefaultLog.WriteInfo("[RTXInstruments]: Configured device %s", name);
            }
        });
    } catch (std::exception& ex) {
        core::utility::log::Log::DefaultLog.WriteError(
            "[RTXInstruments]: Failed to apply configurations.\n%s", ex.what());
    }
}

void RTXInstruments::StartMeasurement(std::filesystem::path const& output_folder,
    std::vector<power::writer_func_t> const& writer_funcs, power::MetaData const* meta, char& signal) {
    if (!std::filesystem::is_directory(output_folder)) {
        core::utility::log::Log::DefaultLog.WriteError(
            "[RTXInstruments]: Path {} is not a directory", output_folder.string());
        return;
    }

    auto t_func = [&](std::filesystem::path const& output_folder, std::vector<power::writer_func_t> const& writer_funcs,
                      power::MetaData const* meta, char& signal) {
        try {
            signal = true;

            waiting_on_trigger();

            int global_device_counter = 0;
            for (auto& [name, i] : rtx_instr_) {
                i.on_operation_complete_ex(
                    [&global_device_counter, num_dev = rtx_instr_.size(), trigger = trigger_.get()](visa_instrument&) {
                        ++global_device_counter;
                        if (global_device_counter >= num_dev) {
                            trigger->DisarmTrigger();
                        }
                    });
                i.acquisition(oscilloscope_acquisition_state::single);
                i.operation_complete_async();
            }

            // magic sleep to wait until devices are ready to recieve other requests
            while (!waiting_on_trigger()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            auto const [trg_prefix, trg_postfix, trg_wait] = get_trigger_timings(config_range_);
            trigger_->ArmTrigger();
            auto tp_fut = std::async(std::launch::async,
                std::bind(&Trigger::StartTriggerSequence, trigger_.get(), trg_prefix, trg_postfix, trg_wait));
            tp_fut.wait();
            auto const last_trigger_ft = tp_fut.get();

            core::utility::log::Log::DefaultLog.WriteInfo("[RTXInstruments]: Start fetching data");
            auto const start_fetch = std::chrono::steady_clock::now();
            std::vector<std::future<oscilloscope_sample>> all_wf_fut;
            all_wf_fut.reserve(rtx_instr_.size());

            for (auto& [name, i] : rtx_instr_) {
                all_wf_fut.push_back(std::async(std::launch::async,
                    std::bind(static_cast<oscilloscope_sample (rtx_instrument::*)(oscilloscope_waveform_points const,
                                  rtx_instrument::timeout_type const)>(&rtx_instrument::data),
                        std::addressof(i), oscilloscope_waveform_points::maximum, 1000)));
            }

            for (auto& f : all_wf_fut) {
                f.wait();
            }
            auto const stop_fetch = std::chrono::steady_clock::now();
            core::utility::log::Log::DefaultLog.WriteInfo("[RTXInstruments]: Stop fetching data %dms",
                std::chrono::duration_cast<std::chrono::milliseconds>(stop_fetch - start_fetch).count());

            std::size_t f_cnt = 0;
            std::for_each(rtx_instr_.begin(), rtx_instr_.end(), [&](auto& instr) {
                auto const& name = instr.first;
                auto& i = instr.second;

                auto fit = rtx_config_.find(name);
                if (fit == rtx_config_.end()) {
                    core::utility::log::Log::DefaultLog.WriteError(
                        "[RTXInstruments]: Could not find config for device {}", name);
                    return;
                }

                auto const& config = fit->second;
                auto const num_channels = config.channels();
                if (num_channels == 0) {
                    core::utility::log::Log::DefaultLog.WriteError("[RTXInstruments]: No configured channels");
                    return;
                }
                std::vector<oscilloscope_channel> channels(num_channels);
                config.channels(channels.data(), channels.size());

                core::utility::log::Log::DefaultLog.WriteInfo("[RTXInstruments]: Start reading data");
                auto const start_read = std::chrono::steady_clock::now();
                auto const all_waveforms = all_wf_fut[f_cnt++].get();

                auto const num_segments = i.history_segments();

                power::segments_t values_map(num_segments);

                for (std::decay_t<decltype(num_segments)> s_idx = 0, fetch_idx = num_segments - 1; s_idx < num_segments;
                     ++s_idx, --fetch_idx) {
                    auto const& waveform = all_waveforms[fetch_idx * num_channels].waveform();
                    auto const sample_times = generate_timeline(waveform);
                    auto const segment_times =
                        offset_timeline(sample_times, std::chrono::duration_cast<power::filetime_dur_t>(
                                                          std::chrono::duration<float>(waveform.segment_offset())));
                    auto const timestamps = offset_timeline(segment_times, last_trigger_ft);
                    values_map[s_idx][global_ts_name] = timestamps;

                    for (unsigned int c_idx = 0; c_idx < num_channels; ++c_idx) {
                        auto const tpn = channels[c_idx].label().text();
                        values_map[s_idx][tpn] =
                            copy_waveform(all_waveforms[fetch_idx * num_channels + c_idx].waveform());
                    }
                }
                auto const stop_read = std::chrono::steady_clock::now();
                core::utility::log::Log::DefaultLog.WriteInfo("[RTXInstruments]: Finished reading data in %dms",
                    std::chrono::duration_cast<std::chrono::milliseconds>(stop_read - start_read).count());
                core::utility::log::Log::DefaultLog.WriteInfo("[RTXInstruments]: Start writing data");
                auto const start_write = std::chrono::steady_clock::now();
                std::for_each(writer_funcs.begin(), writer_funcs.end(),
                    [&output_folder, &name, &values_map, &meta](
                        auto const& writer_func) { writer_func(output_folder, name, values_map, meta); });
                auto const stop_write = std::chrono::steady_clock::now();
                core::utility::log::Log::DefaultLog.WriteInfo("[RTXInstruments]: Finished writing data in %dms",
                    std::chrono::duration_cast<std::chrono::milliseconds>(stop_write - start_write).count());
            });
            signal = false;
        } catch (std::exception& ex) {
            core::utility::log::Log::DefaultLog.WriteError(
                "[RTXInstruments]: Failed to take measurement.\n%s", ex.what());
        }
    };

    auto t_thread = std::thread(t_func, output_folder, writer_funcs, meta, std::ref(signal));
    t_thread.detach();
}

bool RTXInstruments::waiting_on_trigger() const {
    for (auto& [name, i] : rtx_instr_) {
        auto res = i.query("STAT:OPER:COND?\n");
        *strchr(res.as<char>(), '\n') = 0;
        auto const val = std::atoi(res.as<char>());
        if (val & 8) {
            return true;
        }
    }
    return false;
}

} // namespace megamol::power

#endif
