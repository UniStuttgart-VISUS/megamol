#pragma once

#include <list>
#include <string>

namespace megamol::power {
class StringContainer {
public:
    StringContainer() = default;

    StringContainer(StringContainer&&) = default;

    StringContainer& operator=(StringContainer&&) = default;

    std::string const* Add(std::string const& str);

private:
    std::list<std::string> cont_;
};
} // namespace megamol::power
