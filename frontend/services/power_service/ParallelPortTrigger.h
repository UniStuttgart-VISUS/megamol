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

    DWORD Write(const void* data, const DWORD cnt) const;

    DWORD WriteHigh(void) const;

    DWORD WriteLow(void) const;

private:
#ifdef WIN32
    wil::unique_hfile handle_;
#endif
};

} // namespace megamol::frontend

#endif
