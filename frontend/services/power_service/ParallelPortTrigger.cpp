#include "ParallelPortTrigger.h"

#ifdef MEGAMOL_USE_POWER

#include <stdexcept>
#include <system_error>

namespace megamol::frontend {

ParallelPortTrigger::ParallelPortTrigger(char const* path) {
    this->Open(path);
}


void ParallelPortTrigger::Open(char const* path) {
#ifdef WIN32
    this->handle_.reset(::CreateFileA(path, GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL));
    if (!this->handle_) {
        throw std::system_error(::GetLastError(), std::system_category());
    }
#endif
}


DWORD ParallelPortTrigger::Write(const void* data, const DWORD cnt) const {
    DWORD retval = 0;

#ifdef WIN32
    if (!::WriteFile(this->handle_.get(), data, cnt, &retval, nullptr)) {
        throw std::system_error(::GetLastError(), std::system_category());
    }
#endif

    return retval;
}


DWORD ParallelPortTrigger::WriteHigh(void) const {
    static const auto data = (std::numeric_limits<std::uint8_t>::max)();
    return this->Write(&data, sizeof(data));
}


DWORD ParallelPortTrigger::WriteLow(void) const {
    static const char data = 0;
    return this->Write(&data, sizeof(data));
}

} // namespace megamol::frontend

#endif
