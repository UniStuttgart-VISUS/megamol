/**
 * MegaMol
 * Copyright (c) 2016, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/CallGeneric.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore_gl/flags/FlagCollectionGL.h"

namespace megamol::core_gl {

class FlagCallRead_GL : public core::GenericVersionedCall<std::shared_ptr<FlagCollection_GL>, core::EmptyMetaData> {
public:
    inline FlagCallRead_GL() = default;
    ~FlagCallRead_GL() override = default;

    static const char* ClassName() {
        return "FlagCallRead_GL";
    }
    static const char* Description() {
        return "Call that transports a buffer object representing a FlagStorage in a shader storage buffer for reading";
    }
};

class FlagCallWrite_GL : public core::GenericVersionedCall<std::shared_ptr<FlagCollection_GL>, core::EmptyMetaData> {
public:
    inline FlagCallWrite_GL() = default;
    ~FlagCallWrite_GL() override = default;

    static const char* ClassName() {
        return "FlagCallWrite_GL";
    }
    static const char* Description() {
        return "Call that transports a buffer object representing a FlagStorage in a shader storage buffer for writing";
    }
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<FlagCallRead_GL> FlagCallRead_GLDescription;
typedef megamol::core::factories::CallAutoDescription<FlagCallWrite_GL> FlagCallWrite_GLDescription;

} // namespace megamol::core_gl
