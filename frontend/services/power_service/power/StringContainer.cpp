#include "StringContainer.h"

namespace megamol::power {
std::string const* StringContainer::Add(std::string const& str) {
    cont_.push_back(str);
    return &cont_.back();
}
} // namespace megamol::power
