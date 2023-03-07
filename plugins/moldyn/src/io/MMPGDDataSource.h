/*
 * MMPGDDataSource.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AnimDataModule.h"
#include "moldyn/ParticleGridDataCall.h"
#include "vislib/RawStorage.h"
#include "vislib/math/Cuboid.h"
#include "vislib/sys/File.h"
#include "vislib/types.h"


namespace megamol::moldyn::io {

using namespace megamol::core;

/**
 * Data source module for MMPGD files
 */
class MMPGDDataSource : public view::AnimDataModule {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "MMPGDDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Data source module for MMPGD files.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    MMPGDDataSource();

    /** Dtor. */
    ~MMPGDDataSource() override;

protected:
    /**
     * Creates a frame to be used in the frame cache. This method will be
     * called from within 'initFrameCache'.
     *
     * @return The newly created frame object.
     */
    view::AnimDataModule::Frame* constructFrame() const override;

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
    void loadFrame(view::AnimDataModule::Frame* frame, unsigned int idx) override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

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
        ~Frame() override;

        /**
         * Clears the loaded data
         */
        inline void Clear() {
            this->dat.EnforceSize(0);
        }

        /**
         * Loads a frame from 'file' into this object
         *
         * @param file The file stream to load from. The stream is assumed
         *             to be at the correct location
         * @param idx The zero-based index of the frame
         * @param size The size of the frame data in bytes
         *
         * @return True on success
         */
        bool LoadFrame(vislib::sys::File* file, unsigned int idx, UINT64 size);

        /**
         * Sets the data into the call
         *
         * @param call The call to receive the data
         */
        void SetData(ParticleGridDataCall& call);

    private:
        /** position data per type */
        vislib::RawStorage dat;

        /** The types */
        ParticleGridDataCall::ParticleType* types;

        /** The cells */
        ParticleGridDataCall::GridCell* cells;
    };

    /**
     * Helper class to unlock frame data when 'CallSimpleSphereData' is
     * used.
     */
    class Unlocker : public ParticleGridDataCall::Unlocker {
    public:
        /**
         * Ctor.
         *
         * @param frame The frame to unlock
         */
        Unlocker(Frame& frame) : ParticleGridDataCall::Unlocker(), frame(&frame) {
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

    /** The slot for requesting data */
    CalleeSlot getData;

    /** The opened data file */
    vislib::sys::File* file;

    /** The frame index table */
    UINT64* frameIdx;

    /** The data set bounding box */
    vislib::math::Cuboid<float> bbox;

    /** The data set clipping box */
    vislib::math::Cuboid<float> clipbox;
};

} // namespace megamol::moldyn::io
