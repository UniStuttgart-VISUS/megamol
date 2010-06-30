/*
 * AbstractTriMeshDataSource.h
 *
 * Copyright (C) 2010 by Sebastian Grottel
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMTRISOUPPLG_ABSTRACTTRIMESHDATASOURCE_H_INCLUDED
#define MMTRISOUPPLG_ABSTRACTTRIMESHDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Module.h"
#include "CalleeSlot.h"
#include "param/ParamSlot.h"
#include "CallTriMeshData.h"
#include "vislib/Array.h"
#include "vislib/Cuboid.h"
#include "vislib/String.h"


namespace megamol {
namespace trisoup {


    /**
     * Abstract base class for tri-mesh data source classes
     */
    class AbstractTriMeshDataSource : public core::Module {
    public:

        /** Ctor */
        AbstractTriMeshDataSource(void);

        /** Dtor */
        virtual ~AbstractTriMeshDataSource(void);

    protected:

        /** Alias for mesh class */
        typedef CallTriMeshData::Mesh Mesh;

        /** Alias for material class */
        typedef CallTriMeshData::Material Material;

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

        /**
         * Loads the specified file
         *
         * @param filename The file to load
         *
         * @return True on success
         */
        virtual bool load(const vislib::TString& filename) = 0;

        /** The objects */
        vislib::Array<Mesh> objs;

        /** The materials */
        vislib::Array<Material> mats;

        /** The bounding box */
        vislib::math::Cuboid<float> bbox;

    private:

        /** Ensures that the data is loaded */
        inline void assertData(void);

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getDataCallback(core::Call& caller);

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getExtentCallback(core::Call& caller);

        /** The file name */
        core::param::ParamSlot filenameSlot;

        /** The slot for requesting data */
        core::CalleeSlot getDataSlot;

        /** The data update hash */
        SIZE_T datahash;

    };

} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MMTRISOUPPLG_ABSTRACTTRIMESHDATASOURCE_H_INCLUDED */
