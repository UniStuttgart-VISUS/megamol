/*
 * BezierLines.h
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_BEZTUBE_BEZIERLINES_H_INCLUDED
#define MEGAMOL_BEZTUBE_BEZIERLINES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmstd_trisoup/LinesDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/RawStorage.h"
#include "mmcore/misc/BezierCurvesListDataCall.h"


namespace megamol {
namespace beztube {

    /**
     * (Core) lines of bezier data
     */
    class BezierLines : public core::Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "BezierLines";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Generator for line data of bezier curves";
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
        BezierLines(void);

        /** Dtor. */
        virtual ~BezierLines(void);

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
        void makeLines(core::misc::BezierCurvesListDataCall& dat);

        /** The call for data */
        core::CalleeSlot dataSlot;

        /** The call for bezier data */
        core::CallerSlot getDataSlot;

        /** The vertex data */
        vislib::RawStorage vertData;

        /** The index data */
        vislib::RawStorage idxData;

        /** The index data */
        vislib::RawStorage colData;

        /** The data hash */
        SIZE_T inHash;

        /** The data hash */
        SIZE_T outHash;

        /** The frame id */
        unsigned int frameId;

        /** The lines data */
        trisoup::LinesDataCall::Lines lines;

        /** Number of linear segments */
        core::param::ParamSlot numSegsSlot;

    };

} /* end namespace beztube */
} /* end namespace megamol */

#endif /* MEGAMOL_BEZTUBE_BEZIERLINES_H_INCLUDED */
