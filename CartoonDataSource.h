/*
 * CartoonDataSource.h
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_CARTOONDATASOURCE_H_INCLUDED
#define MMPROTEINPLUGIN_CARTOONDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

//#include "misc/ExtBezierDataCall.h"
#include "misc/BezierDataCall.h"
#include "Module.h"
#include "param/ParamSlot.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "MolecularDataCall.h"
#include "Stride.h"
#include "vislib/BezierCurve.h"

namespace megamol {
namespace protein {

    /**
     * Data source for PDB files
     */

    class CartoonDataSource : public megamol::core::Module
    {
    public:

        /** Ctor */
        CartoonDataSource(void);

        /** Dtor */
        virtual ~CartoonDataSource(void);

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void)  {
            return "CartoonDataSource";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers Cartoon input data.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Call callback to get the data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getData( core::Call& call);

        /**
         * Call callback to get the extent of the data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getExtent( core::Call& call);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

        /**
         * Loads a PDB file using PDBLoader.
         *
         * @param filename The input filename.
         *
         * @return 'true' if data could be queried, 'false' otherwise.
         */
        bool loadFile( const vislib::TString& filename);

        /**
         * Compute the bezier points.
         *
         * @param mol The molecular data call.
         */
        void ComputeBezierPoints( const MolecularDataCall *mol);

        /**
         * Compute the bezier points using only tubes.
         *
         * @param mol The molecular data call.
         */
        void ComputeBezierPointsTubes( const MolecularDataCall *mol);

        // -------------------- variables --------------------

        /** The file name slot */
        core::param::ParamSlot filenameSlot;
        /** The data callee slot */
        core::CalleeSlot dataOutSlot;

        /** molecular data caller slot */
        megamol::core::CallerSlot molDataCallerSlot;

        /** The STRIDE usage flag slot */
        core::param::ParamSlot strideFlagSlot;

        /** The data hash */
        SIZE_T datahash;

        ///** The curves data */
        //vislib::Array<vislib::math::BezierCurve<
        //    core::misc::ExtBezierDataCall::Point, 3> > ellipCurves;

        ///** The curves data */
        //vislib::Array<vislib::math::BezierCurve<
        //    core::misc::ExtBezierDataCall::Point, 3> > rectCurves;

        /** The curves data */
        vislib::Array<vislib::math::BezierCurve<
            core::misc::BezierDataCall::BezierPoint, 3> > tubeCurves;

    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_CARTOONDATASOURCE_H_INCLUDED
