#pragma once

#include <cstddef>

namespace megamol::power {
class ProtectedMemory {
public:
    ProtectedMemory(std::size_t const req_size);

    ~ProtectedMemory();

    ProtectedMemory(ProtectedMemory const&) = delete;

    ProtectedMemory& operator=(ProtectedMemory const&) = delete;

    ProtectedMemory(ProtectedMemory&& rhs) noexcept;

    ProtectedMemory& operator=(ProtectedMemory&& rhs) noexcept;

    char* GetPtr() {
        return data_;
    }

    char const* GetPtr() const {
        return data_;
    }

private:
    char* data_ = nullptr;

    std::size_t data_size_ = 0;
};
}
