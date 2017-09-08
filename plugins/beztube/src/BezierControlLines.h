/*
 * BezierControlLines.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_BEZTUBE_BEZIERCONTROLLINES_H_INCLUDED
#define MEGAMOL_BEZTUBE_BEZIERCONTROLLINES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmstd_trisoup/LinesDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/RawStorage.h"
#include "v1/BezierDataCall.h"
#include "mmcore/misc/BezierCurvesListDataCall.h"


namespace megamol {
namespace beztube {

    /**
     * Mesh-based renderer for bézier curve tubes
     */
    class BezierControlLines : public core::Module {
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

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
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
        bool getDataCallback(core::Call& caller);

        /**
         * Get the extent of the line data
         *
         * @param caller The calling caller
         *
         * @return The return value
         */
        bool getExtentCallback(core::Call& caller);

        /**
         * Creates the line data from the input data
         *
         * @param dat The call transporting the input data
         */
        void makeLines(v1::BezierDataCall& dat);

        /**
         * Creates the line data from the input data
         *
         * @param dat The call transporting the input data
         */
        void makeLines(core::misc::BezierCurvesListDataCall& dat);

        /** The call for data */
        core::CalleeSlot dataSlot;

        /** The call for bezier data */
        core::CallerSlot getDataSlot;

        /** The vertex data */
        vislib::RawStorage vertData[2];

        /** The index data */
        vislib::RawStorage idxData[2];

        /** The data hash */
        SIZE_T hash;

        /** The frame id */
        unsigned int frameId;

        /** The lines data */
        trisoup::LinesDataCall::Lines lines[2];

    };

} /* end namespace beztube */
} /* end namespace megamol */

#endif /* MEGAMOL_BEZTUBE_BEZIERCONTROLLINES_H_INCLUDED */
