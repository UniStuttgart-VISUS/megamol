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

#include "mmcore/misc/BezierCurvesListDataCall.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/Array.h"
#include "vislib/math/BezierCurve.h"
#include "vislib/math/Cuboid.h"


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

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
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

        /**
         * Adds a spline from input positions to the stored curves
         *
         * @param pos The input positions (x, y, z)
         * @param times The input time steps (t1, t2)
         * @param cnt The number of input positions
         * @param rad The base radius value
         * @param colR The red colour component of the input points
         * @param colG The green colour component of the input points
         * @param colB The blue colour component of the input points
         */
        void addSpline(float *pos, float *times, unsigned int cnt, float rad, unsigned char colR, unsigned char colG, unsigned char colB);

        /**
         * Answer the colour value for a specific time from the time colour map
         *
         * @param time The time to return the colour for
         * @param outR The variable to receive the red colour value
         * @param outG The variable to recieve the green colour value
         * @param outB The variable to recieve the blue colour value
         */
        void timeColour(float time, unsigned char &outR, unsigned char &outG, unsigned char &outB);

        /** The slot for requesting data */
        CalleeSlot getDataSlot;

        /** The slot for fetching siff data */
        CallerSlot inDataSlot;

        /** Parameter slot defining the colour map */
        param::ParamSlot colourMapSlot;

        /** Parameter slot to compensate cyclic boundary conditions */
        param::ParamSlot deCycleSlot;

        /** The bounding box of positions*/
        vislib::math::Cuboid<float> bbox;

        /** The clipping box of positions */
        vislib::math::Cuboid<float> cbox;

        /** The curves data */
        vislib::Array<misc::BezierCurvesListDataCall::Curves> curves;

        /** The hash value of the outgoing data */
        SIZE_T datahash;

        /** The hash value of the incoming data */
        SIZE_T inhash;

    };

} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SIFFCSPLINEFITTER_H_INCLUDED */
