/*
 * FlagCall_GL.h
 *
 * Author: Guido Reina and others
 * Copyright (C) 2016-2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include "vislib/graphics/gl/IncludeAllGL.h"

#include "FlagStorage.h"
#include "glowl/BufferObject.hpp"
#include "mmcore/CallGeneric.h"
#include "mmcore/UniFlagStorage.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol {
namespace core {

    /**
     * Call for passing flag data (FlagStorage) between modules. This GL
     * variant just passes a GL buffer ID around and allows a call to ensure
     * the storage fits (at least) the required size
     */

    class MEGAMOLCORE_API FlagCallRead_GL
            : public core::GenericVersionedCall<std::shared_ptr<FlagCollection_GL>, core::EmptyMetaData> {
    public:
        inline FlagCallRead_GL() = default;
        ~FlagCallRead_GL() = default;

        static const char* ClassName(void) {
            return "FlagCallRead_GL";
        }
        static const char* Description(void) {
            return "Call that transports a buffer object representing a FlagStorage in a shader storage buffer for "
                   "reading";
        }
    };

    class MEGAMOLCORE_API FlagCallWrite_GL
            : public core::GenericVersionedCall<std::shared_ptr<FlagCollection_GL>, core::EmptyMetaData> {
    public:
        inline FlagCallWrite_GL() = default;
        ~FlagCallWrite_GL() = default;

        static const char* ClassName(void) {
            return "FlagCallWrite_GL";
        }
        static const char* Description(void) {
            return "Call that transports a buffer object representing a FlagStorage in a shader storage buffer for "
                   "writing";
        }
    };

    class FlagCallRead_CPU
            : public core::GenericVersionedCall<std::shared_ptr<FlagCollection_CPU>, core::EmptyMetaData> {
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

    class FlagCallWrite_CPU
            : public core::GenericVersionedCall<std::shared_ptr<FlagCollection_CPU>, core::EmptyMetaData> {
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
    typedef megamol::core::factories::CallAutoDescription<FlagCallRead_GL> FlagCallRead_GLDescription;
    typedef megamol::core::factories::CallAutoDescription<FlagCallWrite_GL> FlagCallWrite_GLDescription;
    typedef megamol::core::factories::CallAutoDescription<FlagCallRead_CPU> FlagCallRead_CPUDescription;
    typedef megamol::core::factories::CallAutoDescription<FlagCallWrite_CPU> FlagCallWrite_CPUDescription;

} // namespace core
} /* end namespace megamol */
