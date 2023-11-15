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

#include "MetaData.h"
#include "ParallelPortTrigger.h"
#include "Trigger.h"
#include "Utility.h"

#include <sol/sol.hpp>

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

#include "mmcore/utility/log/Log.h"

namespace megamol::power {

class RTXInstruments {
public:
    RTXInstruments(std::shared_ptr<Trigger> trigger);

    void UpdateConfigs(std::filesystem::path const& config_folder, int points, int count,
        std::chrono::milliseconds range, std::chrono::milliseconds timeout);

    void ApplyConfigs(MetaData* meta = nullptr);

    void StartMeasurement(std::filesystem::path const& output_folder,
        std::vector<power::writer_func_t> const& writer_funcs, power::MetaData const* meta, char& signal);

    void SetSoftwareTrigger(bool set) {
        enforce_software_trigger_ = set;
        trigger_->RegisterSubTrigger("RTXInstruments", std::bind(&RTXInstruments::soft_trg, this));
    }

private:
    void soft_trg() {
        /*for (auto& [name, i] : rtx_instr_) {
            i.trigger_manually();
        }*/
        main_instr_->trigger_manually();
    }

    bool waiting_on_trigger() const;

    std::unordered_map<std::string, visus::power_overwhelming::rtx_instrument> rtx_instr_;

    visus::power_overwhelming::rtx_instrument* main_instr_ = nullptr;

    std::unordered_map<std::string, visus::power_overwhelming::rtx_instrument_configuration> rtx_config_;

    sol::state sol_state_;

    std::chrono::milliseconds config_range_;

    bool enforce_software_trigger_ = false;

    std::shared_ptr<Trigger> trigger_ = nullptr;
};

} // namespace megamol::power

#endif
