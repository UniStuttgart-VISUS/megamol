#pragma once

#include <cstddef>
#include <string>

#include "ProtectedMemory.h"

namespace megamol::power {
class CryptToken {
public:
    CryptToken(std::string const& filename);

    ~CryptToken() = default;

    char const* GetToken() const;

private:
    power::ProtectedMemory token_;

    std::size_t token_size_;
};
} // namespace megamol::power
