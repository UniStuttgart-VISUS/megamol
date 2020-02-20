/*
 * MMPGDDataSource.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MMPGDDATASOURCE_H_INCLUDED
#define MEGAMOLCORE_MMPGDDATASOURCE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AnimDataModule.h"
#include "rendering/ParticleGridDataCall.h"
#include "vislib/RawStorage.h"
#include "vislib/math/Cuboid.h"
#include "vislib/sys/File.h"
#include "vislib/types.h"


namespace megamol {
namespace stdplugin {
namespace moldyn {
namespace io {

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
    static const char *ClassName(void) { return "MMPGDDataSource"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) { return "Data source module for MMPGD files."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    MMPGDDataSource(void);

    /** Dtor. */
    virtual ~MMPGDDataSource(void);

protected:
    /**
     * Creates a frame to be used in the frame cache. This method will be
     * called from within 'initFrameCache'.
     *
     * @return The newly created frame object.
     */
    virtual view::AnimDataModule::Frame *constructFrame(void) const;

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

private:
    /** Nested class of frame data */
    class Frame : public view::AnimDataModule::Frame {
    public:
        /**
         * Ctor.
         *
         * @param owner The owning AnimDataModule
         */
        Frame(view::AnimDataModule &owner);

        /** Dtor. */
        virtual ~Frame(void);

        /**
         * Clears the loaded data
         */
        inline void Clear(void) { this->dat.EnforceSize(0); }

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
        bool LoadFrame(vislib::sys::File *file, unsigned int idx, UINT64 size);

        /**
         * Sets the data into the call
         *
         * @param call The call to receive the data
         */
        void SetData(rendering::ParticleGridDataCall &call);

    private:
        /** position data per type */
        vislib::RawStorage dat;

        /** The types */
		rendering::ParticleGridDataCall::ParticleType *types;

        /** The cells */
		rendering::ParticleGridDataCall::GridCell *cells;
    };

    /**
     * Helper class to unlock frame data when 'CallSimpleSphereData' is
     * used.
     */
    class Unlocker : public rendering::ParticleGridDataCall::Unlocker {
    public:
        /**
         * Ctor.
         *
         * @param frame The frame to unlock
         */
        Unlocker(Frame &frame)
            : rendering::ParticleGridDataCall::Unlocker()
            , frame(&frame) {
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
    bool filenameChanged(param::ParamSlot &slot);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(Call &caller);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getExtentCallback(Call &caller);

    /** The file name */
    param::ParamSlot filename;

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
};

} // namespace io
} // namespace moldyn
} // namespace stdplugin
} // namespace megamol
#endif /* MEGAMOLCORE_MMPGDDATASOURCE_H_INCLUDED */
