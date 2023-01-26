/*
 * MMSPDDataSource.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MMSPDDATASOURCE_H_INCLUDED
#define MEGAMOLCORE_MMSPDDATASOURCE_H_INCLUDED
#pragma once

#include "geometry_calls/MultiParticleDataCall.h"
#include "io/MMSPDFrameData.h"
#include "io/MMSPDHeader.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AnimDataModule.h"
#include "vislib/RawStorage.h"
#include "vislib/RawStorageWriter.h"
#include "vislib/math/Cuboid.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/sys/Event.h"
#include "vislib/sys/File.h"
#include "vislib/sys/Thread.h"
#include "vislib/types.h"


namespace megamol {
namespace moldyn {
namespace io {


/**
 * Data source module for MMSPD files
 */
class MMSPDDataSource : public core::view::AnimDataModule {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "MMSPDDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Data source module for \"MegaMol Simple Particle Data\" files.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /**
     * Tests if a file can be loaded with this module
     *
     * @param data The data to test
     * @param dataSize The size of the data to test
     *
     * @return The loading confidence value
     */
    static float FileFormatAutoDetect(const unsigned char* data, SIZE_T dataSize);

    /**
     * Answer the file name extensions often used
     *
     * @return The file name extensions
     */
    static const char* FilenameExtensions() {
        return ".mmspd";
    }

    /**
     * Answer the file name slot name
     *
     * @return The file name slot name
     */
    static const char* FilenameSlotName() {
        return "filename";
    }

    /**
     * Answer the file type name (e. g. "Particle Data")
     *
     * @return The file type name
     */
    static const char* FileTypeName() {
        return "MegaMol Simple Particle Data";
    }

    /** Ctor. */
    MMSPDDataSource();

    /** Dtor. */
    ~MMSPDDataSource() override;

protected:
    /**
     * Creates a frame to be used in the frame cache. This method will be
     * called from within 'initFrameCache'.
     *
     * @return The newly created frame object.
     */
    core::view::AnimDataModule::Frame* constructFrame() const override;

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Loads one frame of the data set into the given 'frame' object. This
     * method may be invoked from another thread. You must take
     * precausions in case you need synchronised access to shared
     * ressources.
     *
     * @param frame The frame to be loaded.
     * @param idx The index of the frame to be loaded.
     */
    void loadFrame(core::view::AnimDataModule::Frame* frame, unsigned int idx) override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    /** Nested class of frame data */
    class Frame : public MMSPDFrameData, public core::view::AnimDataModule::Frame {
    public:
        /**
         * Ctor.
         *
         * @param owner The owning AnimDataModule
         */
        Frame(core::view::AnimDataModule& owner);

        /** Dtor. */
        ~Frame() override;

        /**
         * Clears the loaded data
         */
        inline void Clear() {
            this->Data().Clear();
            this->IndexReconstructionData().EnforceSize(0);
        }

        /**
         * Loads a frame from 'file' into this object
         *
         * @param file The file stream to load from. The stream is assumed
         *             to be at the correct location
         * @param idx The zero-based index of the frame
         * @param size The size of the frame data in bytes
         * @param header The data set header
         * @param isBinary Flag whether or not the data set is binary
         * @param isBigEndian Flag whether or not the binary data set is big endian
         *
         * @return True on success
         */
        bool LoadFrame(vislib::sys::File* file, unsigned int idx, UINT64 size, const MMSPDHeader& header, bool isBinary,
            bool isBigEndian);

        /**
         * Sets the data into the call
         *
         * @param call The call to receive the data
         * @param header The data set header
         */
        void SetData(geocalls::MultiParticleDataCall& call, const MMSPDHeader& header);

        /**
         * Sets the directional data into the call
         *
         * @param call The call to receive the data
         * @param header The data set header
         */
        void SetDirData(geocalls::MultiParticleDataCall& call, const MMSPDHeader& header);

    private:
        /**
         * Loads a frame from 'buffer' into this object assuming that
         * 'buffer' holds the data in 7-Bit ASCII form.
         *
         * @param buffer The frame data in main memory
         * @param size The size of 'buffer'
         * @param header The data set header
         *
         * @throws vislib::Exception on any error
         */
        void loadFrameText(char* buffer, UINT64 size, const MMSPDHeader& header);

        /**
         * Loads a frame from 'buffer' into this object assuming that
         * 'buffer' holds the data in binary form.
         *
         * @param buffer The frame data in main memory
         * @param size The size of 'buffer'
         * @param header The data set header
         *
         * @throws vislib::Exception on any error
         */
        void loadFrameBinary(char* buffer, UINT64 size, const MMSPDHeader& header);

        /**
         * Loads a frame from 'buffer' into this object assuming that
         * 'buffer' holds the data in binary (big endian) form.
         *
         * @param buffer The frame data in main memory
         * @param size The size of 'buffer'
         * @param header The data set header
         *
         * @throws vislib::Exception on any error
         */
        void loadFrameBinaryBE(char* buffer, UINT64 size, const MMSPDHeader& header);

        /**
         * Appends a particle of type 'type' to the index-reconstruction data
         *
         * @param type The type of the particle
         * @param wrtr The index data writer
         * @param data The index data store
         * @param lastType The type of the last particle added
         * @param lastCount The number of the last particles of 'lastType' added
         */
        void addIndexForReconstruction(UINT32 type, class vislib::RawStorageWriter& wrtr,
            class vislib::RawStorage& data, UINT32& lastType, UINT64& lastCount);
    };

    /**
     * Helper class to unlock frame data when 'CallSimpleSphereData' is
     * used.
     */
    class Unlocker : public geocalls::MultiParticleDataCall::Unlocker {
    public:
        /**
         * Ctor.
         *
         * @param frame The frame to unlock
         */
        Unlocker(Frame& frame) : geocalls::MultiParticleDataCall::Unlocker(), frame(&frame) {
            // intentionally empty
        }

        /** Dtor. */
        ~Unlocker() override {
            this->Unlock();
            ASSERT(this->frame == NULL);
        }

        /** Unlocks the data */
        void Unlock() override {
            if (this->frame != NULL) {
                this->frame->Unlock();
                this->frame = NULL; // DO NOT DELETE!
            }
        }

    private:
        /** The frame to unlock */
        Frame* frame;
    };

    /**
     * Builds the data file frame index
     *
     * @param userdata Pointer to the calling object
     *
     * @return 0
     */
    static DWORD buildFrameIndex(void* userdata);

    /**
     * Clears the data
     */
    void clearData();

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
     * Gets the directional data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDirDataCallback(core::Call& caller);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getExtentCallback(core::Call& caller);

    /** The file name */
    core::param::ParamSlot filename;

    /** The slot for requesting data */
    core::CalleeSlot getData;

    /** The slot for requesting directional data */
    core::CalleeSlot getDirData;

    /** The data header */
    MMSPDHeader dataHeader;

    /** The opened data file */
    vislib::sys::File* file;

    /**
     * The frame index table, that is the seek positions within the file
     * pointing to the position of the first particle of a frame (directly
     * following the Time Frame Marker).
     * This array is #frames + 1 long to also store the end of the last frame.
     */
    UINT64* frameIdx;

    /** The data set clipping box */
    vislib::math::Cuboid<float> clipbox;

    /** Flag whether or not the data set is a binary file */
    bool isBinaryFile;

    /** Flag whether or not the binary data file uses big endian */
    bool isBigEndian;

    /** The lock for building up the frame index */
    vislib::sys::CriticalSection frameIdxLock;

    /** The event set whenever a new entry gets valid in the frame index */
    vislib::sys::Event frameIdxEvent;

    /** The thread constructing the frame index */
    vislib::sys::Thread frameIdxThread;

    /** The data set data hash */
    SIZE_T dataHash;
};

} /* end namespace io */
} /* end namespace moldyn */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MMSPDDATASOURCE_H_INCLUDED */
