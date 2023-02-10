/*
 * TestSpheresDataSource.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

//#define MMCORE_TEST_DYN_PARAM_SLOTS 1

#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AnimDataModule.h"
#include "vislib/memutils.h"

namespace megamol::moldyn::io {

/**
 * Test data source module providing generated spheres data
 */
class TestSpheresDataSource : public core::view::AnimDataModule {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TestSpheresDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Test data source module providing generated spheres data";
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
    TestSpheresDataSource();

    /** Dtor. */
    ~TestSpheresDataSource() override;

protected:
    /**
     * Creates a frame to be used in the frame cache. This method will be
     * called from within 'initFrameCache'.
     *
     * @return The newly created frame object.
     */
    Frame* constructFrame() const override;

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
    void loadFrame(Frame* frame, unsigned int idx) override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

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

private:
    /**
     * Class storing data of a single frame
     */
    class Frame : public core::view::AnimDataModule::Frame {
    public:
        /**
         * Ctor
         *
         * @param owner The owning module
         */
        Frame(TestSpheresDataSource& owner) : core::view::AnimDataModule::Frame(owner), data(NULL) {
            // intentionally empty
        }

        /**
         * Dtor
         */
        ~Frame() override {
            ARY_SAFE_DELETE(this->data);
        }

        /**
         * Sets the frame number
         *
         * @param n The frame number
         */
        void SetFrameNumber(unsigned int n) {
            this->frame = n;
        }

        /** The particle data */
        float* data;
    };

    /** The slot for requesting data */
    core::CalleeSlot getDataSlot;

    /** Number of frames to be generated */
    core::param::ParamSlot numFramesSlot;

    /** Number of spheres to be generated */
    core::param::ParamSlot numSpheresSlot;

#ifdef MMCORE_TEST_DYN_PARAM_SLOTS
    param::ParamSlot p1;
    param::ParamSlot p2;
#endif
};

} // namespace megamol::moldyn::io
