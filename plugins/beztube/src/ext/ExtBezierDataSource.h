/*
 * ExtBezierDataSource.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef BEZTUBE_EXTBEZIERDATASOURCE_H_INCLUDED
#define BEZTUBE_EXTBEZIERDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "beztube/beztube.h"
#include "ext/ExtBezierDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/Array.h"
#include "vislib/math/BezierCurve.h"
#include "vislib/math/Cuboid.h"


namespace megamol {
namespace beztube {
namespace ext {


    /**
     * Data loader module for 3+1 dim cubic bézier data
     */
    class ExtBezierDataSource : public core::Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ExtBezierDataSource";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Data source module for extended bezier data.";
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
        ExtBezierDataSource(void);

        /** Dtor. */
        virtual ~ExtBezierDataSource(void);

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

        /**
         * Ensures that the data file is loaded into memory, if possible
         */
        void assertData(void);

        /** The file name */
        //core::param::ParamSlot filenameSlot;

        /** The slot for requesting data */
        core::CalleeSlot getDataSlot;

        /** The bounding box of positions*/
        vislib::math::Cuboid<float> bbox;

        /** The curves data */
        vislib::Array<vislib::math::BezierCurve<
            ExtBezierDataCall::Point, 3> > ellipCurves;

        /** The curves data */
        vislib::Array<vislib::math::BezierCurve<
            ExtBezierDataCall::Point, 3> > rectCurves;

        /** The hash value of the loaded data */
        SIZE_T datahash;

    };

} /* end namespace ext */
} /* end namespace beztube */
} /* end namespace megamol */

#endif /* BEZTUBE_EXTBEZIERDATASOURCE_H_INCLUDED */
