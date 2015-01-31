/*
 * BuckyBall.h
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MMSTD_VOLUME_BUCKYBALL_H_INCLUDED
#define MMSTD_VOLUME_BUCKYBALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"


namespace megamol {
namespace stdplugin {
namespace volume {


    /**
     * Class generation buck ball informations
     */
    class BuckyBall : public core::Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "BuckyBall";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Generated Bucky Ball data.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor */
        BuckyBall(void);

        /** Dtor */
        virtual ~BuckyBall(void);

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
        bool getDataCallback(core::Call& caller);

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getExtentCallback(core::Call& caller);

        /** The slot for requesting data */
        core::CalleeSlot getDataSlot;

        /** The distance volume resolution */
        unsigned int volRes;

        /** The distance volume */
        float *vol;

    };

} /* end namespace volume */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MMSTD_VOLUME_BUCKYBALL_H_INCLUDED */
