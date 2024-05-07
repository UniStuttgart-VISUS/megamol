/**
 * MegaMol
 * Copyright (c) 2022, 2023, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "FrameStatistics.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/renderer/RendererModule.h"
#include "mmstd_gl/ModuleGL.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"

namespace megamol::mmstd_gl {

/**
 * Pseudo-Renderer that manipulates time to quickly synchronize data that has only a subset of timesteps
 * with respect to another.
 */
class TimeManipulator : public core::view::RendererModule<CallRender3DGL, ModuleGL> {
public:
    static void requested_lifetime_resources(frontend_resources::ResourceRequest& req) {
        ModuleGL::requested_lifetime_resources(req);
        req.require<frontend_resources::FrameStatistics>();
    }

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TimeManipulator";
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
    TimeManipulator();

    /** Dtor. */
    ~TimeManipulator() override;

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

    bool ManipulateTime(CallRender3DGL& call, CallRender3DGL* chainedCall, uint32_t idx);

    /** Parameter for the time multiplier */
    core::param::ParamSlot multiplierSlot;

    /** Parameters for the length */
    core::param::ParamSlot overrideLengthSlot;
    core::param::ParamSlot resultingLengthSlot;

    core::param::ParamSlot pinTimeSlot;
    core::param::ParamSlot pinnedTimeSlot;

    core::param::ParamSlot showDebugSlot;
    float incomingTime = 0.0f, outgoingTime = 0.0f;
    uint32_t reportedFrameCount = 0;


    uint32_t lastDrawnFrame = std::numeric_limits<uint32_t>::max();
};
} // namespace megamol::mmstd_gl
