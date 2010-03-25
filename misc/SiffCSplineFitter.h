/*
 * SiffCSplineFitter.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_SIFFCSPLINEFITTER_H_INCLUDED
#define MEGAMOLCORE_SIFFCSPLINEFITTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "BezierDataCall.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "Module.h"
#include "param/ParamSlot.h"
#include "vislib/Array.h"
#include "vislib/BezierCurve.h"


namespace megamol {
namespace core {
namespace misc {


    /**
     * Data loader module for 3+1 dim cubic bézier data
     */
    class SiffCSplineFitter : public Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "SiffCSplineFitter";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Module to fit cardinal splines into timed siff data";
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
        SiffCSplineFitter(void);

        /** Dtor. */
        virtual ~SiffCSplineFitter(void);

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
        bool getDataCallback(Call& caller);

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getExtentCallback(Call& caller);

        /**
         * Ensures that the data file is loaded into memory, if possible
         */
        void assertData(void);

        void addSpline(float *pos, float *times, unsigned int cnt, float rad, unsigned char colR, unsigned char colG, unsigned char colB);

        void timeColour(float time, unsigned char &outR, unsigned char &outG, unsigned char &outB);

        /** The slot for requesting data */
        CalleeSlot getDataSlot;

        /** The slot for fetching siff data */
        CallerSlot inDataSlot;

        param::ParamSlot colourMapSlot;

        /** The bounding box of positions*/
        float minX, minY, minZ, maxX, maxY, maxZ;

        /** The curves data */
        vislib::Array<vislib::math::BezierCurve<
            BezierDataCall::BezierPoint, 3> > curves;

        /** The hash value of the outgoing data */
        SIZE_T datahash;

        /** The hash value of the incoming data */
        SIZE_T inhash;

    };

} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SIFFCSPLINEFITTER_H_INCLUDED */
