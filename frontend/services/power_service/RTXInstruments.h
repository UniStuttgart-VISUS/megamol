#pragma once

#ifdef MEGAMOL_USE_POWER

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <power_overwhelming/rtx_instrument.h>
#include <power_overwhelming/rtx_instrument_configuration.h>

#include "Trigger.h"
#include "ParallelPortTrigger.h"
#include "Utility.h"

#include <sol/sol.hpp>

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

#include "mmcore/utility/log/Log.h"

namespace megamol::power {

class RTXInstruments {
public:
    RTXInstruments();

    void UpdateConfigs(std::filesystem::path const& config_folder, int points, int count,
        std::chrono::milliseconds range, std::chrono::milliseconds timeout);

    void ApplyConfigs();

    void StartMeasurement(
        std::filesystem::path const& output_folder, std::vector<power::writer_func_t> const& writer_funcs);

    //void SetLPTTrigger(std::string const& address);

    void SetSoftwareTrigger(bool set) {
        enforce_software_trigger_ = set;
        trigger_->RegisterSubTrigger("RTXInstruments", std::bind(&RTXInstruments::soft_trg, this));
    }

    bool IsMeasurementPending() const {
        return pending_measurement_;
    }

private:
    void soft_trg() {
        for (auto& [name, i] : rtx_instr_) {
            i.trigger_manually();
        }
    }

    bool waiting_on_trigger() const;

    /*std::tuple<std::chrono::system_clock::time_point, int64_t> trigger() {
#ifdef MEGAMOL_USE_TRACY
        ZoneScopedNC("RTXInstruments::trigger", 0xDB0ABF);
#endif
        if (enforce_software_trigger_) {
            for (auto& [name, i] : rtx_instr_) {
                i.trigger_manually();
            }
        } else {
            if (lpt_trigger_) {
                lpt_trigger_->SetBit(6, true);
                lpt_trigger_->SetBit(6, false);
            }
        }

        return std::make_tuple(std::chrono::system_clock::now(), get_highres_timer());
    }*/

    std::unordered_map<std::string, visus::power_overwhelming::rtx_instrument> rtx_instr_;

    std::unordered_map<std::string, visus::power_overwhelming::rtx_instrument_configuration> rtx_config_;

    sol::state sol_state_;

    std::chrono::milliseconds config_range_;

    //std::unique_ptr<ParallelPortTrigger> lpt_trigger_ = nullptr;

    bool enforce_software_trigger_ = false;

    bool pending_measurement_ = false;

    std::unique_ptr<Trigger> trigger_;
};

} // namespace megamol::frontend

#endif
