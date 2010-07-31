/*
 * CallBinaryVolumeData.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMTRISOUPPLG_CALLBINARYVOLUMEDATAA_H_INCLUDED
#define MMTRISOUPPLG_CALLBINARYVOLUMEDATAA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractGetData3DCall.h"
#include "CallAutoDescription.h"
#include "vislib/assert.h"
#include "vislib/Array.h"
#include "vislib/String.h"


namespace megamol {
namespace trisoup {

    /**
     * Call transporting binary volume data
     */
    class CallBinaryVolumeData : public core::AbstractGetData3DCall {
    public:

        /**
         * Answer the name of the objects of this description.
         *
         * @return The name of the objects of this description.
         */
        static const char *ClassName(void) {
            return "CallBinaryVolumeData";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Call transporting binary volume data";
        }

        /**
         * Answer the number of functions used for this call.
         *
         * @return The number of functions used for this call.
         */
        static unsigned int FunctionCount(void) {
            return AbstractGetData3DCall::FunctionCount();
        }

        /**
         * Answer the name of the function used for this call.
         *
         * @param idx The index of the function to return it's name.
         *
         * @return The name of the requested function.
         */
        static const char * FunctionName(unsigned int idx) {
            return AbstractGetData3DCall::FunctionName(idx);
        }

        /** Ctor */
        CallBinaryVolumeData(void);

        /** Dtor */
        virtual ~CallBinaryVolumeData(void);

    private:

    };

    /** Description class typedef */
    typedef core::CallAutoDescription<CallBinaryVolumeData> CallBinaryVolumeDataDescription;

} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MMTRISOUPPLG_CALLBINARYVOLUMEDATAA_H_INCLUDED */
