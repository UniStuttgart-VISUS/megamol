#pragma once

#ifdef MEGAMOL_USE_POWER

#include <atomic>
#include <cstdint>

#ifdef WIN32
#include <wil/resource.h>
#endif

namespace megamol::power {

/**
 * @brief Class representing a trigger device over the parallel port.
 * Produces signals for external hardware sensors.
 */
class ParallelPortTrigger final {
public:
    /**
     * @brief Ctor.
     */
    ParallelPortTrigger() = default;

    /**
     * @brief Ctor. Opens the specified parallel port.
     * If parallel port cannot be opened, @c handle_ remains nullptr.
     * @param path Path of the parallel port.
     */
    explicit ParallelPortTrigger(char const* path);

    /**
     * @brief Opens the specified parallel port.
     * If parallel port cannot be opened, @c handle_ remains nullptr.
     * Resets existing handle.
     * @param path Path of the parallel port.
     */
    void Open(char const* path);

    //DWORD Write(const void* data, const DWORD cnt) const;

    /**
     * @brief Writes data on opened parallel port.
     * @param data The data to write.
     * @return API-specific return code value.
     * @throws std::system_error If write fails.
     */
    uint32_t Write(std::uint8_t data);

    /**
     * @brief Write high on all output bits.
     * @return API-specific return code value.
     * @throws std::system_error If write fails.
     */
    uint32_t WriteHigh();

    /**
     * @brief Write low on all output bits.
     * @return API-specific return code value.
     * @throws std::system_error If write fails.
     */
    uint32_t WriteLow();

    /**
     * @brief Sets specified bit to the value in @c set.
     * @param idx The index of the bit to write.
     * @param state The value to write on the bit.
     * @return API-specific return code value.
     * @throws std::system_error If write fails.
     */
    uint32_t SetBit(unsigned char idx, bool state);

private:
#ifdef WIN32
    wil::unique_hfile handle_;
#endif

    std::atomic<std::uint8_t> data_state_;
};

} // namespace megamol::power

#endif
