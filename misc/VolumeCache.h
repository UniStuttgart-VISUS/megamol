/*
 * VolumeCache.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VOLUMECACHE_H_INCLUDED
#define MEGAMOLCORE_VOLUMECACHE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "param/ParamSlot.h"
#include "Module.h"
#include "vislib/Array.h"
#include "vislib/RawStorage.h"
#include "CallVolumeData.h"
#include "BoundingBoxes.h"


namespace megamol {
namespace core {
namespace misc {


    /**
     * Test data source module providing generated spheres data
     */
    class VolumeCache : public Module {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "VolumeCache";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Caches volume data in an external file";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return true
         */
        static bool IsAvailable(void) {
            return true;
        }

        /**
         * Answer whether or not this module supports being used in a
         * quickstart.
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
        }

        /** Ctor. */
        VolumeCache(void);

        /** Dtor. */
        virtual ~VolumeCache(void);

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

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool outDataCallback(Call& caller);

        /**
         * Gets the data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool outExtentCallback(Call& caller);

    private:

        /**
         * Loads the cache
         *
         * @return True on success
         */
        bool loadCache(void);

        /**
         * Saves the cache
         */
        void saveCache(void);

        /**
         * Rebuilds the cache
         *
         * @param inDat the input data
         */
        void buildCache(CallVolumeData* inDat);

        /** The slot for requesting data */
        CalleeSlot outDataSlot;

        /** The slot getting the data */
        CallerSlot inDataSlot;

        /** The slot for the cache file name */
        param::ParamSlot filenameSlot;

        /** The slot to use the cache file name */
        param::ParamSlot useCacheSlot;

        /** The slot to force an update of the cache and the cache file */
        param::ParamSlot forceAndSaveSlot;

        /** The slot to save the cache and the cache file */
        param::ParamSlot saveSlot;

        /** The data hash */
        SIZE_T dataHash;

        /** The frame index */
        unsigned int frameIdx;

        /** The volume resolution */
        unsigned int res[3];

        /** The volume attribute data */
        vislib::Array<CallVolumeData::Data> attr;

        /** The raw volume data */
        vislib::RawStorage data;

        /** The bounding boxes */
        BoundingBoxes bboxes;

        /** The number of frames */
        unsigned int frameCount;

    };

} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_TESTSPHERESDATASOURCE_H_INCLUDED */
