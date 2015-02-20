/*
 * MolecularBezierData.h
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_PROTEIN_MOLECULARBEZIERDATA_H_INCLUDED
#define MEGAMOL_PROTEIN_MOLECULARBEZIERDATA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/misc/BezierCurvesListDataCall.h"
#include "MolecularDataCall.h"
#include "vislib/Array.h"
#include "vislib/math/Vector.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace protein {

    /**
     * Mesh-based renderer for bézier curve tubes
     */
    class MolecularBezierData : public core::Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "MolecularBezierData";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Converts from 'MolecularDataCall' to 'BezierCurvesListDataCall'";
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
        MolecularBezierData(void);

        /** Dtor. */
        virtual ~MolecularBezierData(void);

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
        void update(MolecularDataCall& dat);

        /** The call for data */
        core::CalleeSlot outDataSlot;

        /** The call for data */
        core::CallerSlot inDataSlot;

        /** The data hash */
        SIZE_T hash;

        /** The data hash */
        SIZE_T outhash;

        /** The time code */
        unsigned int timeCode;

        /** The data */
        core::misc::BezierCurvesListDataCall::Curves data;

        vislib::Array<vislib::math::Vector<float, 3> > colorLookupTable;
        vislib::Array<float> atomColorTable;
        vislib::Array<vislib::math::Vector<float, 3> > rainbowColors;

        core::param::ParamSlot color1Slot;
        core::param::ParamSlot color2Slot;
        core::param::ParamSlot minGradColorSlot;
        core::param::ParamSlot mixGradColorSlot;
        core::param::ParamSlot maxGradColorSlot;
        core::param::ParamSlot colorMixSlot;

    };

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MEGAMOL_PROTEIN_MOLECULARBEZIERDATA_H_INCLUDED */
