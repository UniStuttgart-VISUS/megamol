/*
 * BezDatOpt.h
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_BEZTUBE_BEZDATOPT_H_INCLUDED
#define MEGAMOL_BEZTUBE_BEZDATOPT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/misc/BezierCurvesListDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/Array.h"


namespace megamol {
namespace beztube {


    /**
     * Loader for BezDat files
     *
     * @remarks The whole file needs to fit into memory and is loaded at the
     *          time it is requested for the first time after the filename
     *          parameter has been changed
     */
    class BezDatOpt : public core::Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "BezierDataOptimize";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Optimizes data in 'BezierCurvesListDataCall'";
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

        /** Ctor */
        BezDatOpt(void);

        /** Dtor */
        virtual ~BezDatOpt(void);

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

        /** Utility type to store full point information */
        typedef struct _pt_t {
            float x, y, z, r;
            unsigned char col[3];
            unsigned char counter;
        } full_point_type;

        /**
         * Test of equality for point structs
         *
         * @param lhs The left hand side operand
         * @param rhs The right hand side operand
         * @param epsilon The comparison epsilon
         *
         * @return True if lhs and rhs are equal
         */
        static bool is_equal(const full_point_type& lsh, const full_point_type& rhs, const float epsilon);

        /**
         * Asks for the data
         *
         * @param call The calling call
         *
         * @return True on success
         */
        bool getData(megamol::core::Call& call);

        /**
         * Asks for the extent of the data
         *
         * @param call The calling call
         *
         * @return True on success
         */
        bool getExtent(megamol::core::Call& call);

        /**
         * Ensures that the data is loaded, if possible
         *
         * @param frameID the time frame id of the new data
         */
        void assertData(unsigned int frameID);

        /**
         * Optimizes one curves list
         *
         * @param optDat Will receive the optimized data
         * @param inDat The input data
         */
        void optimize(core::misc::BezierCurvesListDataCall::Curves& optDat,
            const core::misc::BezierCurvesListDataCall::Curves& inDat);

        /**
         * Optimizes the data layout type
         *
         * @param out_layout The optimal data layout type
         * @param out_glob_rad The global radius
         * @param out_glob_col The global colour
         * @param inDat The input data
         */
        void opt_layout(core::misc::BezierCurvesListDataCall::DataLayout& out_layout,
            float& out_glob_rad, unsigned char *out_glob_col,
            const core::misc::BezierCurvesListDataCall::Curves& inDat);

        /** Slot providing data */
        core::CalleeSlot outDataSlot;

        /** Slot fetching data */
        core::CallerSlot inDataSlot;

        /** The data hash */
        SIZE_T dataHash;

        /** The frame id */
        unsigned int frameID;

        /** The optimized curves data */
        vislib::Array<core::misc::BezierCurvesListDataCall::Curves> data;

    };


} /* end namespace beztube */
} /* end namespace megamol */

#endif /* MEGAMOL_BEZTUBE_BEZDATOPT_H_INCLUDED */
