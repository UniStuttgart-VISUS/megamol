/*
 * AbstractTriMeshLoader.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMTRISOUPPLG_ABSTRACTTRIMESHLOADER_H_INCLUDED
#define MMTRISOUPPLG_ABSTRACTTRIMESHLOADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractTriMeshDataSource.h"
#include "geometry_calls_gl/CallTriMeshDataGL.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib/math/Cuboid.h"


namespace megamol {
namespace trisoup_gl {


/**
 * Abstract base class for tri-mesh data source classes
 */
class AbstractTriMeshLoader : public AbstractTriMeshDataSource {
public:
    /** Ctor */
    AbstractTriMeshLoader(void);

    /** Dtor */
    ~AbstractTriMeshLoader(void) override;

protected:
    /**
     * Loads the specified file
     *
     * @param filename The file to load
     *
     * @return True on success
     */
    virtual bool load(const vislib::TString& filename) = 0;

    /** Ensures that the data is loaded */
    void assertData(void) override;

private:
    /** The file name */
    core::param::ParamSlot filenameSlot;
};

} // namespace trisoup_gl
} /* end namespace megamol */

#endif /* MMTRISOUPPLG_ABSTRACTTRIMESHLOADER_H_INCLUDED */
