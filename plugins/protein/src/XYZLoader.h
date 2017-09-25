/*
 * XYZLoader.h
 *
 * Copyright (C) 2011 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_XYZLOADER_H_INCLUDED
#define MMPROTEINPLUGIN_XYZLOADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CalleeSlot.h"
#include "vislib/Array.h"
#include "vislib/math/Vector.h"
#include "vislib/math/Cuboid.h"
#include <fstream>



namespace megamol {
namespace protein {

    /**
     * Data source for XYZ files
     */
    class XYZLoader : public megamol::core::Module
    {
    public:

        /** Ctor */
        XYZLoader(void);

        /** Dtor */
        virtual ~XYZLoader(void);

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void)  {
            return "XYZLoader";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Loads molecule data (XYZ files).";
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
         * Loads a XYZ file.
         *
         * @param filename The path to the file to load.
         */
        void loadFile( const vislib::TString& filename, bool doElectrostatics = true);

        /**
         * Get the radius of the element
         */
        float getElementRadius( vislib::StringA name);

        /**
         * Get the color of the element
         */
        vislib::math::Vector<float, 3> getElementColor( vislib::StringA name);

        // -------------------- variables --------------------

        /** The file name slot */
        core::param::ParamSlot filenameSlot;
        /** The data callee slot */
        core::CalleeSlot dataOutSlot;

        /** The bounding box */
        vislib::math::Cuboid<float> bbox;

        /** The data hash */
        SIZE_T datahash;

        /** The particle count */
        unsigned int particleCount;

        /** The particle positions and radii */
        float *particles;

        /** The particle colors */
        float *colors;

        /** The particle charges */
        float *charges;
    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_XYZLOADER_H_INCLUDED
