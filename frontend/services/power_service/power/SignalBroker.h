#pragma once

#include <algorithm>
#include <vector>

namespace megamol::power {
class SignalBroker {
public:
    char& Get(bool initial) {
        signals_.push_back(initial);
        return signals_.back();
    }

    bool GetValue() const {
        return std::any_of(signals_.begin(), signals_.end(), [](auto const& val) { return val; });
    }

    void Reset() {
        signals_.clear();
    }

private:
    std::vector<char> signals_;
};
} // namespace megamol::power
