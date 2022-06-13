/**
 * MegaMol
 * Copyright (c) 2016, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/factories/CallAutoDescription.h"
#include "mmstd/flags/FlagCollection.h"
#include "mmstd/generic/CallGeneric.h"

namespace megamol::core {

class FlagCallRead_CPU : public GenericVersionedCall<std::shared_ptr<FlagCollection_CPU>, EmptyMetaData> {
public:
    inline FlagCallRead_CPU() = default;
    ~FlagCallRead_CPU() override = default;

    static const char* ClassName() {
        return "FlagCallRead_CPU";
    }
    static const char* Description() {
        return "Call that transports a buffer object representing a FlagStorage in a shader storage buffer for reading";
    }
};

class FlagCallWrite_CPU : public GenericVersionedCall<std::shared_ptr<FlagCollection_CPU>, EmptyMetaData> {
public:
    inline FlagCallWrite_CPU() = default;
    ~FlagCallWrite_CPU() override = default;

    static const char* ClassName() {
        return "FlagCallWrite_CPU";
    }
    static const char* Description() {
        return "Call that transports a buffer object representing a FlagStorage in a shader storage buffer for writing";
    }
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<FlagCallRead_CPU> FlagCallRead_CPUDescription;
typedef megamol::core::factories::CallAutoDescription<FlagCallWrite_CPU> FlagCallWrite_CPUDescription;

} // namespace megamol::core
