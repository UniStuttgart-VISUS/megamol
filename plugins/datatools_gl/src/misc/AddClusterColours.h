/*
 * AddClusterColours.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/RawStorage.h"


namespace megamol::datatools_gl::misc {

/**
 * Renderer for gridded imposters
 */
class AddClusterColours : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "AddClusterColours";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Adds particle colours from a transfer function to the memory stored particles.";
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
    AddClusterColours(void);

    /** Dtor. */
    ~AddClusterColours(void) override;

private:
    /**
     * Utility class used to unlock the additional colour data
     */
    class Unlocker : public geocalls::MultiParticleDataCall::Unlocker {
    public:
        /**
         * ctor.
         *
         * @param inner The inner unlocker object
         */
        Unlocker(geocalls::MultiParticleDataCall::Unlocker* inner);

        /** dtor. */
        ~Unlocker(void) override;

        /** Unlocks the data */
        void Unlock(void) override;

    private:
        /** the inner unlocker */
        geocalls::MultiParticleDataCall::Unlocker* inner;
    };

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void) override;

    /**
     * Implementation of 'Release'.
     */
    void release(void) override;

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

    /** The call for the output data */
    core::CalleeSlot putDataSlot;

    /** The call for the input data */
    core::CallerSlot getDataSlot;

    /** The call for Transfer function */
    core::CallerSlot getTFSlot;

    /** Button to force rebuild of colour data */
    core::param::ParamSlot rebuildButtonSlot;

    /** The last frame */
    unsigned int lastFrame;

    /** The generated colour data */
    vislib::RawStorage colData;

    /** The update hash */
    vislib::RawStorage updateHash;
};


} // namespace megamol::datatools_gl::misc
