#pragma once

#ifdef MEGAMOL_USE_POWER

#include <chrono>
#include <functional>
#include <memory>
#include <regex>
#include <thread>

#include "ParallelPortTrigger.h"
#include "Utility.h"

#ifdef MEGAMOL_USE_TRACY
#include <tracy/Tracy.hpp>
#endif

namespace megamol::power {
/// <summary>
/// Class containing the trigger base functionality to trigger an oscilloscope over the parallel port
/// and synchronize the measurement with other sensors.
/// </summary>
class Trigger final {
public:
    /// <summary>
    /// Ctr setting the address of the parallel port to use.
    /// Expected format: "lpt1" without prefix "\\.\"
    /// </summary>
    /// <param name="address">LPT address</param>
    Trigger(std::string const& address) {
        set_lpt(address);
    }

    /// <summary>
    /// Arms the trigger sequence.
    /// </summary>
    void ArmTrigger() {
        armed_ = true;
    }

    /// <summary>
    /// Disarms the trigger sequence.
    /// Trigger sequence will exit after full cycle.
    /// </summary>
    void DisarmTrigger() {
        armed_ = false;
    }

    /// <summary>
    /// Runs the trigger sequence as long as <c>armed_</c> is <c>true</c>.
    /// The sequence fires a pre trigger first, for instance, to start recording non-osci sensors, since an osci records the time window of the past.
    /// Then the main trigger fires that will start the measurement at the oscilloscope.
    /// Afterwards, a post trigger is fired , for instance, to stop recording of non-osci sensors.
    /// After exit of this trigger sequence a final trigger is fired, for instance, to write the recorded buffers of the measurements.
    /// This function locks access to the parallel port handle.
    /// </summary>
    /// <param name="prefix"></param>
    /// <param name="postfix"></param>
    /// <param name="wait"></param>
    /// <returns>The timestamp of the last main trigger in filetime.</returns>
    filetime_dur_t StartTriggerSequence(std::chrono::milliseconds const& prefix,
        std::chrono::milliseconds const& postfix, std::chrono::milliseconds const& wait) {
        filetime_dur_t trg_tp;
        std::unique_lock<std::mutex> trg_lock(trg_mtx_);
        while (armed_) {
            fire_pre_trigger();
            std::this_thread::sleep_for(prefix);
            trg_tp = fire_trigger();
            std::this_thread::sleep_for(postfix);
            fire_post_trigger();
            std::this_thread::sleep_for(wait);
        }
        fire_fin_trigger();
        return trg_tp;
    }

    void RegisterSignal(std::string const& name, std::function<void(filetime_dur_t const&)> const& signal) {
        signals_[name] = signal;
    }

    void RegisterSubTrigger(std::string const& name, std::function<void()> const& trigger) {
        sub_trigger_[name] = trigger;
    }

    /// <summary>
    ///
    /// </summary>
    /// <param name="name"></param>
    /// <param name="pre_trigger"></param>
    void RegisterPreTrigger(std::string const& name, std::function<void()> const& pre_trigger) {
        pre_trigger_[name] = pre_trigger;
    }

    /// <summary>
    ///
    /// </summary>
    /// <param name="name"></param>
    /// <param name="post_trigger"></param>
    void RegisterPostTrigger(std::string const& name, std::function<void()> const& post_trigger) {
        post_trigger_[name] = post_trigger;
    }

    /// <summary>
    /// Register a final trigger that is fired after the trigger sequence.
    /// </summary>
    /// <param name="name">Name of the trigger. Should be unique.</param>
    /// <param name="fin_trigger">The instance of the function to register.</param>
    void RegisterFinTrigger(std::string const& name, std::function<void()> const& fin_trigger) {
        fin_trigger_[name] = fin_trigger;
    }

    /// <summary>
    ///
    /// </summary>
    /// <param name="address"></param>
    void SetLPTAddress(std::string const& address) {
        set_lpt(address);
    }

    ParallelPortTrigger* GetHandle() {
        return trigger_.get();
    }

private:
    void set_lpt(std::string const& address) {
        std::regex p("^(lpt|LPT)(\\d)$");
        std::smatch m;
        if (!std::regex_search(address, m, p)) {
            throw std::runtime_error("LPT address malformed");
        }

        std::unique_lock<std::mutex> trg_lock(trg_mtx_);
        try {
            trigger_ = std::make_unique<ParallelPortTrigger>(("\\\\.\\" + address).c_str());
        } catch (...) {
            trigger_ = nullptr;
        }
    }

    filetime_dur_t fire_trigger() {
#ifdef MEGAMOL_USE_TRACY
        ZoneScopedNC("Trigger::trigger", 0xDB0ABF);
#endif
        fire_sub_trigger();

        if (trigger_) {
            trigger_->SetBit(6, true);
            trigger_->SetBit(6, false);
        }


        auto const ts = get_highres_timer();
        notify_all(ts);

        return ts;
    }

    void notify_all(filetime_dur_t const& ts) const {
        for (auto const& [n, s] : signals_) {
            s(ts);
        }
    }

    void fire_sub_trigger() const {
#ifdef MEGAMOL_USE_TRACY
        ZoneScopedN("Trigger::fire_sub_trigger");
#endif
        for (auto const& [n, t] : sub_trigger_) {
            t();
        }
    }

    void fire_pre_trigger() const {
#ifdef MEGAMOL_USE_TRACY
        ZoneScopedN("Trigger::fire_pre_trigger");
#endif
        for (auto const& [n, p] : pre_trigger_) {
            p();
        }
    }

    void fire_post_trigger() const {
#ifdef MEGAMOL_USE_TRACY
        ZoneScopedN("Trigger::fire_post_trigger");
#endif
        for (auto const& [n, p] : post_trigger_) {
            p();
        }
    }

    void fire_fin_trigger() const {
#ifdef MEGAMOL_USE_TRACY
        ZoneScopedN("Trigger::fire_fin_trigger");
#endif
        for (auto const& [n, p] : fin_trigger_) {
            p();
        }
    }

    std::unique_ptr<ParallelPortTrigger> trigger_;

    std::unordered_map<std::string, std::function<void(filetime_dur_t const&)>> signals_;

    std::unordered_map<std::string, std::function<void()>> sub_trigger_;

    std::unordered_map<std::string, std::function<void()>> pre_trigger_;

    std::unordered_map<std::string, std::function<void()>> post_trigger_;

    std::unordered_map<std::string, std::function<void()>> fin_trigger_;

    bool armed_ = false;

    std::mutex trg_mtx_;
};
} // namespace megamol::power

#endif
