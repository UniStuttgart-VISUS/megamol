#pragma once

#ifdef MEGAMOL_USE_POWER

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>

#include "ParallelPortTrigger.h"
#include "Timestamp.h"

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
    /// <param name="prefix">Runup time for measurement.</param>
    /// <param name="postfix">Time between trigger and post trigger operations.</param>
    /// <param name="wait">Wait time at the end of trigger sequence.</param>
    /// <returns>The timestamp of the last main trigger in filetime.</returns>
    filetime_dur_t StartTriggerSequence(std::chrono::milliseconds const& prefix,
        std::chrono::milliseconds const& postfix, std::chrono::milliseconds const& wait) {
        filetime_dur_t trg_tp{0};
        std::unique_lock<std::mutex> trg_lock(trg_mtx_);
        fire_init_trigger();
        while (armed_) {
            fire_pre_trigger();
            std::this_thread::sleep_for(prefix + std::chrono::seconds(1)); //< additional second for NVML runup
            trg_tp = fire_trigger();
            std::this_thread::sleep_for(postfix);
            fire_post_trigger();
            std::this_thread::sleep_for(wait);
        }
        fire_fin_trigger();
        return trg_tp;
    }

    /// <summary>
    /// Registers a signal. Signals will be notified when trigger is fired.
    /// </summary>
    /// <param name="name">Name of the signal.</param>
    /// <param name="signal">Function to be called when trigger is fired.</param>
    void RegisterSignal(std::string const& name, std::function<void(filetime_dur_t const&)> const& signal) {
        signals_[name] = signal;
    }

    /// <summary>
    /// Registers a function that is called before a trigger sequence is called.
    /// </summary>
    /// <param name="name">Name of the function.</param>
    /// <param name="trigger">The callable function.</param>
    void RegisterInitTrigger(std::string const& name, std::function<void()> const& trigger) {
        init_trigger_[name] = trigger;
    }

    /// <summary>
    /// Registers a function that is called with the main trigger.
    /// </summary>
    /// <param name="name">Name of the function.</param>
    /// <param name="trigger">The callable function.</param>
    void RegisterSubTrigger(std::string const& name, std::function<void()> const& trigger) {
        sub_trigger_[name] = trigger;
    }

    /// <summary>
    /// Registers a function that is called at the beggining of the prefix time in the trigger sequence.
    /// </summary>
    /// <param name="name">Name of the function.</param>
    /// <param name="pre_trigger">The callable function.</param>
    void RegisterPreTrigger(std::string const& name, std::function<void()> const& pre_trigger) {
        pre_trigger_[name] = pre_trigger;
    }

    /// <summary>
    /// Registers a function that is called at the end of the postfix time in the trigger sequence.
    /// </summary>
    /// <param name="name">Name of the function.</param>
    /// <param name="post_trigger">The callable function.</param>
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
    /// Opens the parallel port used for sending the main trigger.
    /// </summary>
    /// <param name="address">Address of the virtual file representing the parallel port.</param>
    void SetLPTAddress(std::string const& address) {
        set_lpt(address);
    }

    /// <summary>
    /// Return a pointer the parallel port used for sending the main trigger.
    /// </summary>
    /// <returns>Pointer to parallel port.</returns>
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

        auto const ts = get_highres_timer();
        if (trigger_) {
            trigger_->SetBit(6, true);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            trigger_->SetBit(6, false);
        }

        notify_all(ts);

        return ts;
    }

    void notify_all(filetime_dur_t const& ts) const {
        for (auto const& [n, s] : signals_) {
            s(ts);
        }
    }

    void fire_init_trigger() const {
#ifdef MEGAMOL_USE_TRACY
        ZoneScopedN("Trigger::fire_init_trigger");
#endif
        for (auto const& [n, t] : init_trigger_) {
            t();
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

    std::unordered_map<std::string, std::function<void()>> init_trigger_;

    std::unordered_map<std::string, std::function<void()>> sub_trigger_;

    std::unordered_map<std::string, std::function<void()>> pre_trigger_;

    std::unordered_map<std::string, std::function<void()>> post_trigger_;

    std::unordered_map<std::string, std::function<void()>> fin_trigger_;

    bool armed_ = false;

    std::mutex trg_mtx_;
};
} // namespace megamol::power

#endif
