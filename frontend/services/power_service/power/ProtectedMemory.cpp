#include "ProtectedMemory.h"

#include <stdexcept>

#if WIN32
#include <Windows.h>
#include <wincrypt.h>
#pragma comment(lib, "Crypt32.lib")
#endif

namespace megamol::power {
ProtectedMemory::ProtectedMemory(std::size_t const req_size) {
#if WIN32
    data_size_ = req_size;
    auto cbMod = req_size % CRYPTPROTECTMEMORY_BLOCK_SIZE;
    if (cbMod) {
        data_size_ += CRYPTPROTECTMEMORY_BLOCK_SIZE - cbMod;
    }
    data_ = (char*) LocalAlloc(LPTR, data_size_);
    if (!data_) {
        data_size_ = 0;
        throw std::runtime_error("[ProtectedMemory] Cannot allocate data");
    }
    if (!CryptProtectMemory(data_, data_size_, CRYPTPROTECTMEMORY_SAME_PROCESS)) {
        LocalFree(data_);
        data_size_ = 0;
        throw std::runtime_error("[ProtectedMemory] Cannot encrypt data");
    }
    SecureZeroMemory(data_, data_size_);
#endif
}

ProtectedMemory::~ProtectedMemory() {
#if WIN32
    if (data_) {
        SecureZeroMemory(data_, data_size_);
        LocalFree(data_);
    }
    data_size_ = 0;
#endif
}

ProtectedMemory::ProtectedMemory(ProtectedMemory&& rhs) noexcept
        : data_(std::exchange(rhs.data_, nullptr))
        , data_size_(std::exchange(rhs.data_size_, 0)) {}

ProtectedMemory& ProtectedMemory::operator=(ProtectedMemory&& rhs) noexcept {
    data_ = std::exchange(rhs.data_, nullptr);
    data_size_ = std::exchange(rhs.data_size_, 0);
    return *this;
}
} // namespace megamol::power
