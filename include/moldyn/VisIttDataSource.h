/*
 * VisIttDataSource.h
 *
 * Copyright (C) 2009-2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VISITTDATASOURCE_H_INCLUDED
#define MEGAMOLCORE_VISITTDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "view/AnimDataModule.h"
#include "param/ParamSlot.h"
#include "CalleeSlot.h"
#include "moldyn/MultiParticleDataCall.h"
#include "vislib/Array.h"
#include "vislib/math/Cuboid.h"
#include "vislib/sys/File.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/Pair.h"
#include "vislib/RawStorage.h"


namespace megamol {
namespace core {
namespace moldyn {



    /**
     * Renderer for rendering the vis logo into the unit cube.
     */
    class VisIttDataSource : public view::AnimDataModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "VisIttDataSource";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Data source module for VisItt files.";
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
        VisIttDataSource(void);

        /** Dtor. */
        virtual ~VisIttDataSource(void);

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
        virtual void loadFrame(view::AnimDataModule::Frame *frame,
            unsigned int idx);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

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
             * Access t othe particle data
             *
             * @return The particle data
             */
            inline vislib::RawStorage& Data(void) {
                return this->dat;
            }

            /**
             * Access t othe particle data
             *
             * @return The particle data
             */
            inline const vislib::RawStorage& Data(void) const {
                return this->dat;
            }

            /**
             * Answer the size of the used memory in bytes
             *
             * @return The size of the used memory
             */
            inline SIZE_T Size(void) const {
                return vislib::math::Min<SIZE_T>(this->size, this->dat.GetSize());
            }

            /**
             * Sets the size of the used memory
             *
             * @param s The size of the used memory
             */
            inline void SetSize(SIZE_T s) {
                this->size = s;
            }

            /**
             * Sets the frame number
             *
             * @param fn The frame number
             */
            inline void SetFrameNumber(unsigned int fn) {
                this->frame = fn;
            }

        private:

            /** The size of the memory really used */
            SIZE_T size;

            /** The xyz particle positions */
            vislib::RawStorage dat;

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

        /** Builds up the frame index table. */
        void buildFrameTable(void);

        ///** Calculates the bounding box from all frames. */
        //void calcBoundingBox(void);

        /**
         * Callback receiving the update of the file name parameter.
         *
         * @param slot The updated ParamSlot.
         *
         * @return Always 'true' to reset the dirty flag.
         */
        bool filenameChanged(param::ParamSlot& slot);

        /**
         * The filter settings changed, so we need to reload all data
         *
         * @param slot The slot
         *
         * @return 'true'
         */
        bool filterChanged(param::ParamSlot& slot);

        /**
         * Parses the file header containing the particle descriptions
         *
         * @param header The file header line
         *
         * @return 'true' on success, 'false' on failure.
         */
        bool parseHeader(const vislib::StringA& header);

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

        /** Finds the data column used for filtering */
        void findFilterColumn(void);

        /** The file name */
        param::ParamSlot filename;

        /** The radius for the particles */
        param::ParamSlot radius;

        /** The filter to be applied */
        param::ParamSlot filter;

        /** The filter column to be applied */
        param::ParamSlot filterColumn;

        /** The filter value to be applied */
        param::ParamSlot filterValue;

        /** The slot for requesting data */
        CalleeSlot getData;

        /** The opened data file */
        vislib::sys::File *file;

        /** The file data hash number */
        SIZE_T dataHash;

        /** The frame seek table */
        vislib::Array<vislib::sys::File::FileSize> frameTable;

        /** All column widths and labels */
        vislib::Array<vislib::Pair<vislib::StringA, unsigned int> > header;

        /** The sorted index of the usable columns */
        vislib::Array<unsigned int> headerIdx;

        /** The column index used for filtering */
        unsigned int filterIndex;

        /** The bounding box */
        vislib::math::Cuboid<float> bbox;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VISITTDATASOURCE_H_INCLUDED */
