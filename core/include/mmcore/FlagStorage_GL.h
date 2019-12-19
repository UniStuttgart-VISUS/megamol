/*
 * FlagStorage_GL.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_FLAGSTORAGEGL_H_INCLUDED
#define MEGAMOL_FLAGSTORAGEGL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <mutex>
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/RawStorage.h"
#include "vislib/math/Cuboid.h"
#include "glowl/BufferObject.hpp"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "FlagStorage.h"
#include "FlagCollection_GL.h"

namespace megamol {
namespace core {

/**
 * Class holding a GL buffer of uints which contain flags that say something
 * about a synchronized other piece of data (index equality).
 * Can be used for storing selection etc. Should be kept in sync with the normal
 * FlagStorage, which resides on CPU.
 */
class MEGAMOLCORE_API FlagStorage_GL : public core::Module {
public:
    //enum { ENABLED = 1 << 0, FILTERED = 1 << 1, SELECTED = 1 << 2, SOFTSELECTED = 1 << 3 };

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "FlagStorage_GL"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Module representing an index-synced array of flag uints as a GPU buffer"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return ogl_IsVersionGEQ(4, 3); }

    /** Ctor. */
    FlagStorage_GL(void);

    /** Dtor. */
    virtual ~FlagStorage_GL(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

private:
    /**
     * Access the flags provided by the FlagStorage_GL
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool readDataCallback(core::Call& caller);

    /**
     * Write/update the flags provided by the FlagStorage_GL
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool writeDataCallback(core::Call& caller);

    /**
     * Access the metadata provided by the FlagStorage_GL
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool readMetaDataCallback(core::Call& caller);

    /**
     * Write/update the metadata provided by the FlagStorage_GL
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool writeMetaDataCallback(core::Call& caller);

    /** The slot for reading the data */
    core::CalleeSlot readFlagsSlot;

    /** The slot for writing the data */
    core::CalleeSlot writeFlagsSlot;

    std::shared_ptr<FlagCollection_GL> theData;
    uint32_t version = 0;
};

} // namespace core
} /* end namespace megamol */

#endif /* MEGAMOL_FLAGSTORAGEGL_H_INCLUDED */
