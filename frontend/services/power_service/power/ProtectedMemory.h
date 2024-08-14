#pragma once

#include <cstddef>

namespace megamol::power {
/**
 * @brief Class representing an allocation of protected memory.
 */
class ProtectedMemory {
public:
    /**
     * @brief Allocate @c req_size (+ padding) bytes of protected memory.
     * @param req_size Bytes to allocate.
     * @throws std::runtime_error If allocation or encryption fails.
     */
    ProtectedMemory(std::size_t const req_size);

    /**
     * @brief Dtor.
     */
    ~ProtectedMemory();

    ProtectedMemory(ProtectedMemory const&) = delete;

    ProtectedMemory& operator=(ProtectedMemory const&) = delete;

    /**
     * @brief Move Ctor.
     */
    ProtectedMemory(ProtectedMemory&& rhs) noexcept;

    /**
     * @brief Move assignment.
     */
    ProtectedMemory& operator=(ProtectedMemory&& rhs) noexcept;

    /**
     * @brief Get pointer to begin of allocation.
     * @return The pointer.
     */
    char* GetPtr() {
        return data_;
    }

    /**
     * @brief Get const pointer to begin of allocation.
     * @return The pointer.
     */
    char const* GetPtr() const {
        return data_;
    }

private:
    char* data_ = nullptr;

    std::size_t data_size_ = 0;
};
} // namespace megamol::power
