#pragma once

#ifdef MEGAMOL_USE_POWER

#include <chrono>
#include <functional>
#include <memory>
#include <regex>

#include "ParallelPortTrigger.h"
#include "Utility.h"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

namespace megamol::power {
class Trigger {
public:
    Trigger(std::string const& address) {
        std::regex p("^(lpt|LPT)(\\d)$");
        std::smatch m;
        if (!std::regex_search(address, m, p)) {
            throw std::runtime_error("LPT address malformed");
        }

        try {
            trigger_ = std::make_unique<ParallelPortTrigger>(("\\\\.\\" + address).c_str());
        } catch (...) {
            trigger_ = nullptr;
        }
    }

    void ArmTrigger() {
        armed_ = true;
    }

    void DisarmTrigger() {
        armed_ = false;
    }

    std::tuple<std::chrono::system_clock::time_point, int64_t> StartTriggerSequence(
        std::chrono::milliseconds const& prefix, std::chrono::milliseconds const& postfix,
        std::chrono::milliseconds const& wait) {
        std::tuple<std::chrono::system_clock::time_point, int64_t> trg_tp;
        while (armed_) {
            fire_pre_trigger();
            std::this_thread::sleep_for(prefix);
            trg_tp = fire_trigger();
            std::this_thread::sleep_for(postfix + wait - prefix);
        }
        return trg_tp;
    }

    void RegisterSignal(std::function<void()> const& signal) {
        signals_.push_back(signal);
    }

    void RegisterSubTrigger(std::function<void()> const& trigger) {
        sub_trigger_.push_back(trigger);
    }

    void RegisterPreTrigger(std::function<void()> const& pre_trigger) {
        pre_trigger_.push_back(pre_trigger);
    }

private:
    std::tuple<std::chrono::system_clock::time_point, int64_t> fire_trigger() {
#ifdef MEGAMOL_USE_TRACY
        ZoneScopedNC("Trigger::trigger", 0xDB0ABF);
#endif
        fire_sub_trigger();

        if (trigger_) {
            trigger_->SetBit(6, true);
            trigger_->SetBit(6, false);
        }

        notify_all();

        return std::make_tuple(std::chrono::system_clock::now(), get_highres_timer());
    }

    void notify_all() const {
        for (auto const& s : signals_) {
            s();
        }
    }

    void fire_sub_trigger() const {
        for (auto const& t : sub_trigger_) {
            t();
        }
    }

    void fire_pre_trigger() const {
        for (auto const& p : pre_trigger_) {
            p();
        }
    }

    std::unique_ptr<ParallelPortTrigger> trigger_;

    std::vector<std::function<void()>> signals_;

    std::vector<std::function<void()>> sub_trigger_;

    std::vector<std::function<void()>> pre_trigger_;

    bool armed_ = false;
};
} // namespace megamol::power

#endif
