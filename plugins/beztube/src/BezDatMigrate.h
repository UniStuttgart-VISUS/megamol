/*
 * BezDatMigrate.h
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_BEZTUBE_BezDatMigrate_H_INCLUDED
#define MEGAMOL_BEZTUBE_BezDatMigrate_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
//#include "vislib/RawStorage.h"
#include "ext/ExtBezierDataCall.h"
#include "v1/BezierDataCall.h"
#include "mmcore/misc/BezierCurvesListDataCall.h"


namespace megamol {
namespace beztube {

    /**
     * Mesh-based renderer for bézier curve tubes
     */
    class BezDatMigrate : public core::Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "BezierDataMigrate";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Converts from 'ExtBezierDataCall' and 'v1.BezierDataCall' to 'BezierCurvesListDataCall'";
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
        BezDatMigrate(void);

        /** Dtor. */
        virtual ~BezDatMigrate(void);

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
         * Updates the curve data from the incoming data
         *
         * @param dat The incoming data
         */
        void update(ext::ExtBezierDataCall& dat);

        /**
         * Updates the curve data from the incoming data
         *
         * @param dat The incoming data
         */
        void update(v1::BezierDataCall& dat);

        /** The call for data */
        core::CalleeSlot outDataSlot;

        /** The call for data */
        core::CallerSlot inDataSlot;

        /** The data hash */
        SIZE_T hash;

        /** The time code */
        unsigned int timeCode;

        /** The data */
        core::misc::BezierCurvesListDataCall::Curves data;

    };

} /* end namespace beztube */
} /* end namespace megamol */

#endif /* MEGAMOL_BEZTUBE_BEZIERCONTROLLINES_H_INCLUDED */
