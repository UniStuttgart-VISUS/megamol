/**
 * MegaMol
 * Copyright (c) 2023, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/platform/TypeInfo.h"

#if __has_include(<cxxabi.h>)
#define MEGAMOL_HAS_CXXABI_H
#include <cxxabi.h>
#endif

std::string megamol::core::utility::platform::DemangleTypeName(const char* name) {
#ifdef MEGAMOL_HAS_CXXABI_H
    std::size_t length = 0;
    int status = 0;
    auto* demangled_name = abi::__cxa_demangle(name, nullptr, &length, &status);

    if (demangled_name != nullptr) {
        std::string result(demangled_name);
        std::free(demangled_name);
        return result;
    }

    return std::string(name);
#else
    return std::string(name);
#endif
}
