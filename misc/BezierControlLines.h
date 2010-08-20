/*
 * BezierControlLines.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_BEZIERCONTROLLINES_H_INCLUDED
#define MEGAMOLCORE_BEZIERCONTROLLINES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Module.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "misc/LinesDataCall.h"
#include "param/ParamSlot.h"
#include "vislib/RawStorage.h"


namespace megamol {
namespace core {
namespace misc {

    /**
     * Mesh-based renderer for bézier curve tubes
     */
    class BezierControlLines : public Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "BezierControlLines";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Generator for line data of bezier curve control points";
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
        BezierControlLines(void);

        /** Dtor. */
        virtual ~BezierControlLines(void);

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
         * Get the line data
         *
         * @param caller The calling caller
         *
         * @return The return value
         */
        bool getDataCallback(Call& caller);

        /**
         * Get the extent of the line data
         *
         * @param caller The calling caller
         *
         * @return The return value
         */
        bool getExtentCallback(Call& caller);

        /** The call for data */
        CalleeSlot dataSlot;

        /** The call for bezier data */
        CallerSlot getDataSlot;

        /** The vertex data */
        vislib::RawStorage vertData[2];

        /** The index data */
        vislib::RawStorage idxData[2];

        /** The data hash */
        SIZE_T hash;

        /** The lines data */
        LinesDataCall::Lines lines[2];

    };

} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_BEZIERCONTROLLINES_H_INCLUDED */
