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
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_trisoup/CallTriMeshData.h"
#include "vislib/Array.h"
#include "vislib/math/Cuboid.h"
#include "vislib/String.h"


namespace megamol {
namespace trisoup {


    /**
     * Abstract base class for tri-mesh data source classes
     */
    class AbstractTriMeshLoader : public AbstractTriMeshDataSource {
    public:

        /** Ctor */
        AbstractTriMeshLoader(void);

        /** Dtor */
        virtual ~AbstractTriMeshLoader(void);

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
        virtual void assertData(void);

    private:

        /** The file name */
        core::param::ParamSlot filenameSlot;

    };

} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MMTRISOUPPLG_ABSTRACTTRIMESHLOADER_H_INCLUDED */
