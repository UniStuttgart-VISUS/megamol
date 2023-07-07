#pragma once

#ifdef MEGAMOL_USE_POWER

#include <cinttypes>
#include <limits>

#ifdef WIN32
#include <wil/resource.h>
#endif

namespace megamol::frontend {

class ParallelPortTrigger final {

public:
    ParallelPortTrigger(void) = default;

    explicit ParallelPortTrigger(char const* path);

    void Open(char const* path);

    //DWORD Write(const void* data, const DWORD cnt) const;
    DWORD Write(std::uint8_t data);

    DWORD WriteHigh(void);

    DWORD WriteLow(void);

    DWORD SetBit(unsigned char idx, bool state);

private:
#ifdef WIN32
    wil::unique_hfile handle_;
#endif

    std::uint8_t data_state_;
};

} // namespace megamol::frontend

#endif
