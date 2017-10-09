/*
 * CaverTunnelResidueLoader.h
 * Copyright (C) 2006-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "mmcore/view/AnimDataModule.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/sys/File.h"

namespace megamol {
namespace sombreros {

	/**
	 * Data source for the tunnel-parallel residue files from the Caver software
	 */
	class CaverTunnelResidueLoader : public core::view::AnimDataModule {
	public:

		/**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "CaverTunnelResidueLoader";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Data source module for tunnel-residing residue index files outputted by Caver";
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
		CaverTunnelResidueLoader(void);

		/** Dtor. */
		virtual ~CaverTunnelResidueLoader(void);

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
        virtual void loadFrame(core::view::AnimDataModule::Frame *frame, unsigned int idx);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

		/** Nested class of frame data */
        class Frame : public core::view::AnimDataModule::Frame {
        public:

            /**
             * Ctor.
             *
             * @param owner The owning AnimDataModule
             */
            //TODOFrame(core::view::AnimDataModule& owner);

            /** Dtor. */
            //TODOvirtual ~Frame(void);

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
            //TODObool LoadFrame(vislib::sys::File *file, unsigned int idx, UINT64 size, unsigned int version);

            /**
             * Sets the data into the call
             *
             * @param call The call to receive the data
             */
            //TODOvoid SetData(MultiParticleDataCall& call);

        private:

            /** position data per type */
            vislib::RawStorage dat;

            /** file version */
            unsigned int fileVersion;

        };
#if 0
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
#endif

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

		core::param::ParamSlot filenameSlot;

		core::CalleeSlot getData;

		vislib::sys::File * file;

		size_t data_hash;
	};

}
}