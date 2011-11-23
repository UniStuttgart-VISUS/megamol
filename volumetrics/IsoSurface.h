/*
 * IsoSurface.h
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ISOSURFACE_H_INCLUDED
#define MEGAMOLCORE_ISOSURFACE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Module.h"
#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "param/ParamSlot.h"
#include "vislib/Cuboid.h"
#include "vislib/RawStorage.h"
#include "CallTriMeshData.h"
#include "vislib/Point.h"


namespace vislib {
    class RawStorageWriter;
}

namespace megamol {
namespace trisoup {
namespace volumetrics {


    /**
     * Generator for iso surface tri mesh based on volume data
     */
    class IsoSurface : public core::Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "IsoSurface";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Maps volume data to a ball grid.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor */
        IsoSurface(void);

        /** Dtor */
        virtual ~IsoSurface(void);

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
         * TODO: Reina must also document this ... or better make this a more nicely vislib-function (yoda)
         */
        static float getOffset(float fValue1, float fValue2, float fValueDesired);

        /**
         * Magic table #5
         */
        static const unsigned int tets[6][4];

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool outDataCallback(core::Call& caller);

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool outExtentCallback(core::Call& caller);

        /**
         * Creates the iso surface mesh
         *
         * @param i The index buffer
         * @param v The vertices
         * @param n The normals
         * @param val The iso value
         * @param vol The volume data (scalar)
         * @param sx Sample count in x direction
         * @param sy Sample count in y direction
         * @param sz Sample count in z direction
         */
        void buildMesh(
            vislib::RawStorageWriter& i,
            vislib::RawStorageWriter& v,
            vislib::RawStorageWriter& n,
            float val,
            const float *vol,
            unsigned int sx,
            unsigned int sy,
            unsigned int sz);

        /**
         * Magic Method #12
         *
         * TODO: Reina must document this
         */
        void makeTet(unsigned int triIdx,
            vislib::math::Point<float, 3>* pts,
            float v0,
            float v1,
            float v2,
            float v3,
            float val,
            vislib::RawStorageWriter& idxWrtr,
            vislib::RawStorageWriter& vrtWrtr,
            vislib::RawStorageWriter& nrlWrtr);

        /** The slot for requesting input data */
        core::CallerSlot inDataSlot;

        /** The slot for requesting output data */
        core::CalleeSlot outDataSlot;

        /** The attribute to show */
        core::param::ParamSlot attributeSlot;

        /** The iso value*/
        core::param::ParamSlot isoValueSlot;

        /** The data hash */
        SIZE_T dataHash;

        /** The frame index */
        unsigned int frameIdx;

        /** The object space bounding box */
        vislib::math::Cuboid<float> osbb;

        /** Raw storage holding the index array */
        vislib::RawStorage index;

        /** Raw storage holding the vertex array */
        vislib::RawStorage vertex;

        /** Raw storage holding the normal array */
        vislib::RawStorage normal;

        /** My mesh */
        CallTriMeshData::Mesh mesh;

    };


} /* end namespace volumetrics */
} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ISOSURFACE_H_INCLUDED */
