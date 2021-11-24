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

#include "geometry_calls_gl/CallTriMeshDataGL.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/RawStorage.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/Point.h"


// #define WITH_COLOUR_DATA

namespace vislib {
class RawStorageWriter;
}

namespace megamol {
namespace trisoup_gl {
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
    static const char* ClassName(void) {
        return "IsoSurface";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
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
     * TODO: Document this! Reina!!11
     */
    static vislib::math::Point<float, 3> interpolate(
        vislib::math::Point<float, 3>* pts, float* cv, float val, unsigned int idx0, unsigned int idx1);

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
    void buildMesh(vislib::RawStorageWriter& i, vislib::RawStorageWriter& v,
#ifdef WITH_COLOUR_DATA
        vislib::RawStorageWriter& c,
#endif /* WITH_COLOUR_DATA */
        vislib::RawStorageWriter& n, float val, const float* vol, unsigned int sx, unsigned int sy, unsigned int sz);

    /**
     * Magic Method #12
     *
     * TODO: Reina must document this
     */
    void makeTet(unsigned int triIdx, vislib::math::Point<float, 3>* pts, float v0, float v1, float v2, float v3,
        float val, vislib::RawStorageWriter& idxWrtr, vislib::RawStorageWriter& vrtWrtr,
        vislib::RawStorageWriter& nrlWrtr);

    /**
     * Magic Method #13
     *
     * TODO: Reina must document this
     *
     * @param pts All eight voxel positions
     */
    void makeTet(unsigned int triIdx, unsigned int tetIdx, vislib::math::Point<float, 3>* pts, float* cv, float val,
        vislib::RawStorageWriter& idxWrtr, vislib::RawStorageWriter& vrtWrtr, vislib::RawStorageWriter& nrlWrtr);

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

#ifdef WITH_COLOUR_DATA
    /** Raw storage holding the colour array */
    vislib::RawStorage colour;
#endif /* WITH_COLOUR_DATA */

    /** Raw storage holding the normal array */
    vislib::RawStorage normal;

    /** My mesh */
    megamol::geocalls_gl::CallTriMeshDataGL::Mesh mesh;
};


} /* end namespace volumetrics */
} // namespace trisoup_gl
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ISOSURFACE_H_INCLUDED */
