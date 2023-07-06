/*
 * AbstractTriMeshDataSource.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "geometry_calls/LinesDataCall.h"
#include "geometry_calls_gl/CallTriMeshDataGL.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib/math/Cuboid.h"


namespace megamol::trisoup_gl {


/**
 * Abstract base class for tri-mesh data source classes
 */
class AbstractTriMeshDataSource : public core::Module {
public:
    /** Ctor */
    AbstractTriMeshDataSource();

    /** Dtor */
    ~AbstractTriMeshDataSource() override;

protected:
    /** Alias for mesh class */
    typedef megamol::geocalls_gl::CallTriMeshDataGL::Mesh Mesh;

    /** Alias for material class */
    typedef megamol::geocalls_gl::CallTriMeshDataGL::Material Material;

    /** Alias for lines class */
    typedef megamol::geocalls::LinesDataCall::Lines Lines;

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool getDataCallback(core::Call& caller);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    virtual bool getExtentCallback(core::Call& caller);

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /** Ensures that the data is loaded */
    virtual void assertData() = 0;

    /** The objects */
    vislib::Array<Mesh> objs;

    /** The materials */
    vislib::Array<Material> mats;

    /** The lines */
    std::vector<Lines> lines;

    /** The bounding box */
    vislib::math::Cuboid<float> bbox;

    /** The data update hash */
    SIZE_T datahash;

private:
    /** The slot for requesting data */
    core::CalleeSlot getDataSlot;

    /** The slot for requesting lines data */
    core::CalleeSlot getLinesDataSlot;
};

} // namespace megamol::trisoup_gl
