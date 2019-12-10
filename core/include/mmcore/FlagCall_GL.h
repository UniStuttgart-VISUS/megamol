/*
 * FlagCall_GL.h
 *
 * Author: Guido Reina
 * Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef MEGAMOL_FLAGCALLGL_H_INCLUDED
#define MEGAMOL_FLAGCALLGL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "FlagStorage.h"
#include "mmcore/CallGeneric.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "glowl/BufferObject.hpp"
#include "mmcore/FlagStorage_GL.h"

namespace megamol {
namespace core {

/**
 * Call for passing flag data (FlagStorage) between modules. This GL
 * variant just passes a GL buffer ID around and allows a call to ensure
 * the storage fits (at least) the required size
 */

class MEGAMOLCORE_API FlagCallRead_GL : public core::GenericVersionedCall<std::shared_ptr<FlagCollection_GL>, core::EmptyMetaData> {
public:
    inline FlagCallRead_GL() = default;
    ~FlagCallRead_GL() = default;

    static const char* ClassName(void) { return "FlagCallRead_GL"; }
    static const char* Description(void) { return "Call that transports a buffer object representing a FlagStorage in a shader storage buffer for reading"; }
};

class MEGAMOLCORE_API FlagCallWrite_GL : public core::GenericVersionedCall<std::shared_ptr<FlagCollection_GL>, core::EmptyMetaData> {
public:
    inline FlagCallWrite_GL() = default;
    ~FlagCallWrite_GL() = default;

    static const char* ClassName(void) { return "FlagCallWrite_GL"; }
    static const char* Description(void) { return "Call that transports a buffer object representing a FlagStorage in a shader storage buffer for writing"; }
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<FlagCallRead_GL> FlagCallRead_GLDescription;
typedef megamol::core::factories::CallAutoDescription<FlagCallWrite_GL> FlagCallWrite_GLDescription;

} // namespace core
} /* end namespace megamol */

#endif /* MEGAMOL_FLAGCALLGL_H_INCLUDED */
