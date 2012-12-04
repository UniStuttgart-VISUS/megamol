/*
 * DatRawDataSource.h
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DATRAWDATASOURCE_H_INCLUDED
#define MEGAMOLCORE_DATRAWDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Module.h"
#include "param/ParamSlot.h"
#include "CalleeSlot.h"
#include "vislib/Cuboid.h"
#include "vislib/RawStorage.h"
#include "VolumeDataCall.h"
#include "datRaw.h"


namespace megamol {
namespace core {
namespace moldyn {

    /**
     * Data source for Dat-Raw files (volume data).
     */
    class DatRawDataSource : public Module {
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
        bool filenameChanged(param::ParamSlot& slot);
        
        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getDataCallback(Call& caller);

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getExtentCallback(Call& caller);
        
        /** The dat file name */
        param::ParamSlot datFilenameSlot;

        /** The slot for requesting data */
        CalleeSlot getDataSlot;
        
        /** The bounding box */
        vislib::math::Cuboid<float> bbox;
        
        /** The data */
        vislib::RawStorage data;

        /** The data hash */
        SIZE_T datahash;

        /** The DatRaw file header */
        DatRawFileInfo header;
    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DATRAWDATASOURCE_H_INCLUDED */
