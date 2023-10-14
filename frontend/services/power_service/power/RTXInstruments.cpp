#include "RTXInstruments.h"

#ifdef MEGAMOL_USE_POWER

#include <algorithm>
#include <bitset>
#include <future>
#include <stdexcept>
#include <thread>

#include <mmcore/utility/log/Log.h>

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
        auto name = get_name(i);
        core::utility::log::Log::DefaultLog.WriteInfo("[RTXInstruments]: Found %s as %s", name, get_identity(i));
        rtx_instr_[get_name(i)] = std::move(i);
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

void RTXInstruments::ApplyConfigs() {
    try {
        std::for_each(rtx_instr_.begin(), rtx_instr_.end(), [&](auto& instr) {
            auto const& name = instr.first;
            auto& i = instr.second;
            auto fit = rtx_config_.find(name);
            if (fit == rtx_config_.end()) {
                core::utility::log::Log::DefaultLog.WriteWarn("[RTXInstruments]: No config found for device %s", name);
            } else {
                i.synchronise_clock();
                i.reset(true, true);
                // need to stop running measurement otherwise wait on trigger cannot be used to guard start of trigger sequence
                i.write("STOP\n");
                i.operation_complete();
                fit->second.apply(i);
                i.reference_position(oscilloscope_reference_point::left);
                i.trigger_position(oscilloscope_quantity(0, "ms"));
                i.operation_complete();
                core::utility::log::Log::DefaultLog.WriteInfo("[RTXInstruments]: Configured device %s", name);
            }

            // TODO: store configs in metadata
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
            pending_measurement_ = true;
            signal = true;
            std::chrono::system_clock::time_point last_trigger;
            int64_t last_trigger_qpc;

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
            //std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            while (!waiting_on_trigger()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            trigger_->ArmTrigger();
            auto tp_fut = std::async(std::launch::async,
                std::bind(&Trigger::StartTriggerSequence, trigger_.get(), config_range_ / 12,
                    config_range_ - config_range_ / 12, std::chrono::milliseconds(1000) + config_range_));
            /*while (global_device_counter < rtx_instr_.size()) {
                core::utility::log::Log::DefaultLog.WriteInfo("Waiting on trigger %d", global_device_counter);
                if (waiting_on_trigger()) {
                    core::utility::log::Log::DefaultLog.WriteInfo("Trigger!");
                    std::tie(last_trigger, last_trigger_qpc) = trigger();
                } else {
                    trigger_->DisarmTrigger();
                    break;
                }
                std::this_thread::sleep_for(config_range_ * 1.1f);
            }*/
            /*while (global_device_counter < rtx_instr_.size()) {
                std::this_thread::sleep_for(config_range_ / 2);
            }
            trigger_->DisarmTrigger();*/
            tp_fut.wait();
            std::tie(last_trigger, last_trigger_qpc) = tp_fut.get();

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
                //auto const all_waveforms = i.data(oscilloscope_waveform_points::maximum);
                auto const all_waveforms = all_wf_fut[f_cnt++].get();

                auto const num_segments = i.history_segments();

                power::segments_t values_map(num_segments);

                for (std::decay_t<decltype(num_segments)> s_idx = 0; s_idx < num_segments; ++s_idx) {
                    //auto const fullpath = output_folder / (instr.first + "_s" + std::to_string(s_idx) + ".parquet");
                    auto const& waveform = all_waveforms[s_idx * num_channels].waveform();
                    auto const sample_times = generate_timestamps_ns(waveform);
                    auto const abs_times =
                        offset_timeline(sample_times, std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                          std::chrono::duration<float>(waveform.segment_offset())));
                    auto const ltrg_ns = tp_dur_to_epoch_ns(last_trigger);
                    auto const wall_times = offset_timeline(abs_times, ltrg_ns);
                    power::timeline_t timestamps(abs_times.size());
                    std::transform(abs_times.begin(), abs_times.end(), timestamps.begin(),
                        [&last_trigger_qpc](auto const& base) { return get_tracy_time(base, last_trigger_qpc); });

                    values_map[s_idx]["sample_times"] = sample_times;
                    values_map[s_idx]["abs_times"] = abs_times;
                    values_map[s_idx]["wall_times"] = wall_times;
                    values_map[s_idx]["timestamps"] = timestamps;

                    for (unsigned int c_idx = 0; c_idx < num_channels; ++c_idx) {
                        auto const tpn = name + "_" + channels[c_idx].label().text();
                        values_map[s_idx][tpn] =
                            transform_waveform(all_waveforms[s_idx * num_channels + c_idx].waveform());
                    }

                    //ParquetWriter(fullpath, values_map[s_idx]);
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
            pending_measurement_ = false;
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
        //core::utility::log::Log::DefaultLog.WriteInfo("[RTXInstrument]: Stat Cond: {}", std::bitset<32>(val).to_string());
        if (val & 8) {
            return true;
        }
    }
    return false;
}

} // namespace megamol::power

#endif
