#include "ParallelPortTrigger.h"

#ifdef MEGAMOL_USE_POWER

#include <cstdint>
#include <limits>
#include <system_error>

#include "mmcore/utility/log/Log.h"

namespace megamol::power {

ParallelPortTrigger::ParallelPortTrigger(char const* path) {
    this->Open(path);
}


void ParallelPortTrigger::Open(char const* path) {
#ifdef WIN32
    this->handle_.reset(::CreateFileA(
        path, GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH, NULL));
    if (!this->handle_) {
        //throw std::system_error(::GetLastError(), std::system_category());
        core::utility::log::Log::DefaultLog.WriteWarn("[ParallelPortTriger] Could not open parallel port %s", path);
    }
#endif
}


//DWORD ParallelPortTrigger::Write(const void* data, const DWORD cnt) const {
//    DWORD retval = 0;
//
//#ifdef WIN32
//    if (!::WriteFile(this->handle_.get(), data, cnt, &retval, nullptr)) {
//        throw std::system_error(::GetLastError(), std::system_category());
//    }
//#endif
//
//    return retval;
//}


DWORD ParallelPortTrigger::Write(std::uint8_t data) {
    DWORD retval = 0;

#ifdef WIN32
    if (handle_) {
        if (!::WriteFile(this->handle_.get(), &data, 1, &retval, nullptr)) {
            throw std::system_error(::GetLastError(), std::system_category());
        }
    }
#endif

    return retval;
}


DWORD ParallelPortTrigger::WriteHigh(void) {
    static const auto data = (std::numeric_limits<std::uint8_t>::max)();
    //return this->Write(&data, sizeof(data));
    return Write(data);
}


DWORD ParallelPortTrigger::WriteLow(void) {
    static const char data = 0;
    //return this->Write(&data, sizeof(data));
    return Write(data);
}


std::uint8_t set_bit(std::uint8_t data, unsigned char idx, bool state) {
    std::uint8_t val = state ? 1 : 0;
    return (data & ~(1UL << idx)) | (val << idx);
}


DWORD ParallelPortTrigger::SetBit(unsigned char idx, bool state) {
    if (idx < 8) {
        auto data = data_state_.load();
        while (data_state_.compare_exchange_weak(data, set_bit(data, idx, state))) {}
        return Write(data);
    }

    return 0;
}

} // namespace megamol::power

#endif
