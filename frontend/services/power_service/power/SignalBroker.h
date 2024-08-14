#pragma once

#include <algorithm>
#include <list>

namespace megamol::power {
/// <summary>
/// Helper class to gather and distribute ready/finished signal from different entities.
/// </summary>
class SignalBroker {
public:
    /// <summary>
    /// Created a new signal entry and returns a reference to that entry.
    /// </summary>
    /// <param name="initial">Initial value; 0 false; 1 or higher true.</param>
    /// <returns>Reference to the pushed signal value to set later.</returns>
    char& Get(bool initial) {
        signals_.push_back(initial);
        return signals_.back();
    }

    /// <summary>
    /// Returns true if any of the signals is true.
    /// </summary>
    /// <returns>True, if any of signals is true; false if none of the signals is true.</returns>
    bool GetValue() const {
        return std::any_of(signals_.begin(), signals_.end(), [](auto const& val) { return val; });
    }

    /// <summary>
    /// Clears the list of signals.
    /// </summary>
    void Reset() {
        signals_.clear();
    }

private:
    std::list<char> signals_;
};
} // namespace megamol::power
