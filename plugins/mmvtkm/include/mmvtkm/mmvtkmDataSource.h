/*
 * mmvtkmDataSource.h (MMPLDDataSource)
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_MMVTKM_MMVTKMDATASOURCE_H_INCLUDED
#define MEGAMOL_MMVTKM_MMVTKMDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/AnimDataModule.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmvtkm/mmvtkmDataCall.h"
#include "vislib/math/Cuboid.h"
#include "vislib/sys/File.h"
#include "vislib/RawStorage.h"
#include "vislib/types.h"


namespace megamol {
namespace mmvtkm {



    /**
     * Data source module for mmvtkm files
     */
    class mmvtkmDataSource : public core::view::AnimDataModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "vtkmDataSource";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Data source module for vtkm files.";
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
        mmvtkmDataSource(void);

        /** Dtor. */
        virtual ~mmvtkmDataSource(void);

    protected:
        /**
         * Creates a frame to be used in the frame cache. This method will be
         * called from within 'initFrameCache'.
         *
         * @return The newly created frame object.
         */
        virtual core::view::AnimDataModule::Frame* constructFrame(void) const;

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Loads one frame of the data set into the given 'frame' object. This
         * method may be invoked from another thread. You must take
         * precausions in case you need synchronised access to shared
         * ressources.
         *
         * @param frame The frame to be loaded.
         * @param idx The index of the frame to be loaded.
         */
        virtual void loadFrame(core::view::AnimDataModule::Frame* frame, unsigned int idx);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

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
         * Gets the meta data from the source.
         *
         * @param caller The calling call.
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool getMetaDataCallback(core::Call& caller);

        /** The file name */
        core::param::ParamSlot filename;

        /** The slot for requesting data */
        core::CalleeSlot getData;

        /** The opened data file */
        vislib::sys::File *file;

        /** The data set bounding box */
        vislib::math::Cuboid<float> bbox;

        /** The data set clipping box */
        vislib::math::Cuboid<float> clipbox;

        /** file version */
        unsigned int fileVersion;

        /** Data file load id counter */
        size_t data_hash;

		/** The vtkm data holder */
        vtkm::cont::DataSet vtkmData;

		/** The vtkm data file name */
        std::string vtkmDataFile;

		bool dirtyFlag;
    };

} /* end namespace mmvtkm */
} /* end namespace megamol */

#endif /* MEGAMOL_MMVTKM_MMVTKMDATASOURCE_H_INCLUDED */
