/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/param/ParamSlot.h"
#include "mmstd/renderer/RendererModule.h"
#include "mmstd_gl/ModuleGL.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"

namespace megamol::mmstd_gl {

/**
 * Pseudo-Renderer that manipulates time to quickly synchronize data that has only a subset of timesteps
 * with respect to another.
 */
class TimeDivisor : public core::view::RendererModule<CallRender3DGL, ModuleGL> {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TimeDivisor";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "manipulates time to quickly synchronize data that has only a subset of timesteps";
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
    TimeDivisor();

    /** Dtor. */
    ~TimeDivisor() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    /*
     * Copies the incoming call to the outgoing one to pass the extents
     *
     * @param call The call containing all relevant parameters
     * @return True on success, false otherwise
     */
    bool GetExtents(CallRender3DGL& call) override;

    /*
     * Renders the bounding box and the viewcube on top of the other rendered things
     *
     * @param call The call containing the camera and other parameters
     * @return True on success, false otherwise
     */
    bool Render(CallRender3DGL& call) final;

    /** Parameter for the time divisor */
    core::param::ParamSlot divisorSlot;
};
} // namespace megamol::mmstd_gl
