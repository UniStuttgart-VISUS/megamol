/*
 * FlagStorage.h
 *
 * Copyright (C) 2016 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_FLAGSTORAGE_H_INCLUDED
#define MEGAMOL_FLAGSTORAGE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "vislib/math/Cuboid.h"
#include "vislib/RawStorage.h"
#include "vislib/sys/CriticalSection.h"


namespace megamol {
namespace infovis {



    /**
     * Class storing a stream of uints which contain flags that say something
     * about a synchronized other piece of data (index equality).
     * Can be used for storing selection etc.
     */
    class FlagStorage : public core::Module {
    public:

        enum {
            ENABLED = 1 << 0
            , FILTERED = 1 << 1
            , SELECTED = 1 << 2
            , SOFTSELECTED = 1 << 3
        };

        typedef uint32_t FlagItemType;

        typedef std::vector<FlagItemType> FlagVectorType;

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "FlagStorage";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Module holding an index-synced array of flag uints for other data";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

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
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getFlagsCallback(core::Call& caller);

        /**
         * Sets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool setFlagsCallback(core::Call& caller);

        /** The slot for requesting data */
        core::CalleeSlot getFlagsSlot;

        /** The data */
        std::shared_ptr<const FlagVectorType> flags;

        vislib::sys::CriticalSection crit;

    };

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MEGAMOL_FLAGSTORAGE_H_INCLUDED */
