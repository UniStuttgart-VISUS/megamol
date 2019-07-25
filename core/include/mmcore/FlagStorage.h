/*
 * FlagStorage.h
 *
 * Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_FLAGSTORAGE_H_INCLUDED
#define MEGAMOL_FLAGSTORAGE_H_INCLUDED
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


namespace megamol {
namespace core {


/**
 * Class storing a stream of uints which contain flags that say something
 * about a synchronized other piece of data (index equality).
 * Can be used for storing selection etc.
 */
class MEGAMOLCORE_API FlagStorage : public core::Module {
public:
    enum { ENABLED = 1 << 0, FILTERED = 1 << 1, SELECTED = 1 << 2, SOFTSELECTED = 1 << 3 };

    typedef uint32_t FlagItemType;
    typedef uint32_t FlagVersionType;

    typedef std::vector<FlagItemType> FlagVectorType;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "FlagStorage"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Module holding an index-synced array of flag uints for other data"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    FlagStorage(void);

    /** Dtor. */
    virtual ~FlagStorage(void);

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
     * Gets the data from the source, locking and removing it.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool mapFlagsCallback(core::Call& caller);

    /**
     * Returns the data to the source, version in the call indicating
     * whether it has changed. Then the data is unlocked for other
     * threads to access.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool unmapFlagsCallback(core::Call& caller);

    /** The slot for requesting data */
    core::CalleeSlot getFlagsSlot;

    /** The data */
    std::shared_ptr<FlagVectorType> flags;

    FlagVersionType version;

    // std::recursive_mutex mut;
    std::mutex mut;
};

} // namespace core
} /* end namespace megamol */

#endif /* MEGAMOL_FLAGSTORAGE_H_INCLUDED */
