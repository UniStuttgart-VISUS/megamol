/*
 * MMPLDDataSource.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MMPLDDATASOURCE_H_INCLUDED
#define MEGAMOLCORE_MMPLDDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/AnimDataModule.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "vislib/math/Cuboid.h"
#include "vislib/sys/File.h"
#include "vislib/RawStorage.h"
#include "vislib/types.h"


namespace megamol {
namespace core {
namespace moldyn {



    /**
     * Data source module for MMPLD files
     */
    class MMPLDDataSource : public view::AnimDataModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "MMPLDDataSource";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Data source module for MMPLD files.";
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
        MMPLDDataSource(void);

        /** Dtor. */
        virtual ~MMPLDDataSource(void);

    protected:

        /**
         * Creates a frame to be used in the frame cache. This method will be
         * called from within 'initFrameCache'.
         *
         * @return The newly created frame object.
         */
        virtual view::AnimDataModule::Frame* constructFrame(void) const;

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
        virtual void loadFrame(view::AnimDataModule::Frame *frame, unsigned int idx);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

        /** Nested class of frame data */
        class Frame : public view::AnimDataModule::Frame {
        public:

            /**
             * Ctor.
             *
             * @param owner The owning AnimDataModule
             */
            Frame(view::AnimDataModule& owner);

            /** Dtor. */
            virtual ~Frame(void);

            /**
             * Clears the loaded data
             */
            inline void Clear(void) {
                this->dat.EnforceSize(0);
            }

            /**
             * Loads a frame from 'file' into this object
             *
             * @param file The file stream to load from. The stream is assumed
             *             to be at the correct location
             * @param idx The zero-based index of the frame
             * @param size The size of the frame data in bytes
             * @param version File version (100 = standard, 101 with clusterInfos)
             *
             * @return True on success
             */
            bool LoadFrame(vislib::sys::File *file, unsigned int idx, UINT64 size, unsigned int version);

            /**
             * Sets the data into the call
             *
             * @param call The call to receive the data
             */
            void SetData(MultiParticleDataCall& call, vislib::math::Cuboid<float> const& bbox, bool overrideBBox);

        private:

            /** position data per type */
            vislib::RawStorage dat;

            /** file version */
            unsigned int fileVersion;

        };

        /**
         * Helper class to unlock frame data when 'CallSimpleSphereData' is
         * used.
         */
        class Unlocker : public MultiParticleDataCall::Unlocker {
        public:

            /**
             * Ctor.
             *
             * @param frame The frame to unlock
             */
            Unlocker(Frame& frame) : MultiParticleDataCall::Unlocker(),
                    frame(&frame) {
                // intentionally empty
            }

            /** Dtor. */
            virtual ~Unlocker(void) {
                this->Unlock();
                ASSERT(this->frame == NULL);
            }

            /** Unlocks the data */
            virtual void Unlock(void) {
                if (this->frame != NULL) {
                    this->frame->Unlock();
                    this->frame = NULL; // DO NOT DELETE!
                }
            }

        private:

            /** The frame to unlock */
            Frame *frame;

        };

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

        /** The file name */
        param::ParamSlot filename;

        /** Limits the memory cache size */
        param::ParamSlot limitMemorySlot;

        /** Specifies the size limit of the memory cache */
        param::ParamSlot limitMemorySizeSlot;

        /** Override local bbox */
        param::ParamSlot overrideBBoxSlot;

        /** The slot for requesting data */
        CalleeSlot getData;

        /** The opened data file */
        vislib::sys::File *file;

        /** The frame index table */
        UINT64 *frameIdx;

        /** The data set bounding box */
        vislib::math::Cuboid<float> bbox;

        /** The data set clipping box */
        vislib::math::Cuboid<float> clipbox;

        /** file version */
        unsigned int fileVersion;

        /** Data file load id counter */
        size_t data_hash;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MMPLDDATASOURCE_H_INCLUDED */
