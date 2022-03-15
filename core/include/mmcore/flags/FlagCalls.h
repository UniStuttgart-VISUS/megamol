/*
 * FlagCalls.h
 *
 * Author: Guido Reina and others
 * Copyright (C) 2016-2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include "mmcore/CallGeneric.h"
#include "mmcore/FlagStorage.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol {
namespace core {

class FlagCallRead_CPU : public core::GenericVersionedCall<std::shared_ptr<FlagCollection_CPU>, core::EmptyMetaData> {
public:
    inline FlagCallRead_CPU() = default;
    ~FlagCallRead_CPU() = default;

    static const char* ClassName(void) {
        return "FlagCallRead_CPU";
    }
    static const char* Description(void) {
        return "Call that transports a buffer object representing a FlagStorage in a shader storage buffer for "
               "reading";
    }
};

class FlagCallWrite_CPU : public core::GenericVersionedCall<std::shared_ptr<FlagCollection_CPU>, core::EmptyMetaData> {
public:
    inline FlagCallWrite_CPU() = default;
    ~FlagCallWrite_CPU() = default;

    static const char* ClassName(void) {
        return "FlagCallWrite_CPU";
    }
    static const char* Description(void) {
        return "Call that transports a buffer object representing a FlagStorage in a shader storage buffer for "
               "writing";
    }
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<FlagCallRead_CPU> FlagCallRead_CPUDescription;
typedef megamol::core::factories::CallAutoDescription<FlagCallWrite_CPU> FlagCallWrite_CPUDescription;

} // namespace core
} /* end namespace megamol */
