/*
 * DatRawDataSource.h
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MMSTD_VOLUME_DATRAWDATASOURCE_H_INCLUDED
#define MMSTD_VOLUME_DATRAWDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CalleeSlot.h"
#include "vislib/math/Cuboid.h"
#include "vislib/RawStorage.h"
#include "mmcore/moldyn/VolumeDataCall.h"
#include "datRaw.h"


namespace megamol {
namespace stdplugin {
namespace volume {

    /**
     * Data source for Dat-Raw files (volume data).
     */
    class DatRawDataSource : public core::Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "DatRawDataSource";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Dat-Raw file data source module";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        DatRawDataSource(void);

        /** Dtor. */
        virtual ~DatRawDataSource(void);

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
         * Callback receiving the update of the file name parameter.
         *
         * @param slot The updated ParamSlot.
         *
         * @return Always 'true' to reset the dirty flag.
         */
        bool filenameChanged(core::param::ParamSlot& slot);
        
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
        
        /** The dat file name */
        core::param::ParamSlot datFilenameSlot;

        /** The slot for requesting data */
        core::CalleeSlot getDataSlot;
        
        /** The bounding box */
        vislib::math::Cuboid<float> bbox;
        
        /** The data */
        vislib::RawStorage data;

        /** The data hash */
        SIZE_T datahash;

        /** The DatRaw file header */
        DatRawFileInfo header;
    };

} /* end namespace volume */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MMSTD_VOLUME_DATRAWDATASOURCE_H_INCLUDED */
