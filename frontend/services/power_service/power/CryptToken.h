#pragma once

#include <cstddef>
#include <string>

namespace megamol::power {
class CryptToken {
public:
    CryptToken(std::string const& filename, void* window_ptr = nullptr);

    ~CryptToken();

    char const* GetToken() const;

private:
    char* token_safe_;

    std::size_t token_size_;
};
} // namespace megamol::power
