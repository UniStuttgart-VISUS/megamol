/*
 * BezierDataSource.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_BEZTUBE_BEZIERDATASOURCE_H_INCLUDED
#define MEGAMOL_BEZTUBE_BEZIERDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "v1/BezierDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/Array.h"
#include "vislib/math/BezierCurve.h"


namespace megamol {
namespace beztube {
namespace v1 {


    /**
     * Data loader module for 3+1 dim cubic bézier data
     */
    class BezierDataSource : public core::Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "v1.BezierDataSource";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Data source module for Bezier data.";
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
        BezierDataSource(void);

        /** Dtor. */
        virtual ~BezierDataSource(void);

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

        /**
         * Load a BezDat file into the memory
         *
         * @param filename The path of the file to load
         */
        void loadBezDat(const vislib::TString& filename);

        /** The file name */
        core::param::ParamSlot filenameSlot;

        /** The slot for requesting data */
        core::CalleeSlot getDataSlot;

        /** The bounding box of positions*/
        float minX, minY, minZ, maxX, maxY, maxZ;

        /** The curves data */
        vislib::Array<vislib::math::BezierCurve<
            v1::BezierDataCall::BezierPoint, 3> > curves;

        /** The hash value of the loaded data */
        SIZE_T datahash;

    };

} /* end namespace v1 */
} /* end namespace beztube */
} /* end namespace megamol */

#endif /* MEGAMOL_BEZTUBE_BEZIERDATASOURCE_H_INCLUDED */
