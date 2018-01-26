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

#include "mmcore/Module.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "geometry_calls/CallTriMeshData.h"
#include "geometry_calls/LinesDataCall.h"
#include "vislib/Array.h"
#include "vislib/math/Cuboid.h"
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
        typedef megamol::geocalls::CallTriMeshData::Mesh Mesh;

        /** Alias for material class */
        typedef megamol::geocalls::CallTriMeshData::Material Material;

        /** Alias for lines class */
        typedef megamol::geocalls::LinesDataCall::Lines Lines;

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

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
        virtual void release(void);

        /** Ensures that the data is loaded */
        virtual void assertData(void) = 0;

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

} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MMTRISOUPPLG_ABSTRACTTRIMESHDATASOURCE_H_INCLUDED */
