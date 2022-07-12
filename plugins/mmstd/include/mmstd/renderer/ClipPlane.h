/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/math/Plane.h"

namespace megamol::core::view {

/**
 * Module defining a clipping plane
 */
class ClipPlane : public Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ClipPlane";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Module defining a clipping plane";
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
    ClipPlane();

    /** Dtor. */
    virtual ~ClipPlane();

private:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create();

    /**
     * Implementation of 'Release'.
     */
    virtual void release();

    /**
     * Callback called when the clipping plane is requested.
     *
     * @param call The calling call
     *
     * @return 'true' on success
     */
    bool requestPlane(Call& call);

    /** The callee slot called on request of a clipping plane */
    CalleeSlot getClipPlaneSlot;

    /** The clipping plane */
    vislib::math::Plane<float> plane;

    /** The colour of the plane */
    float col[4];

    /** Disables or enables the clipping plane */
    param::ParamSlot enableSlot;

    /** Defines the colour of the clipping plane */
    param::ParamSlot colourSlot;

    /** Defines the normal of the clipping plane */
    param::ParamSlot normalSlot;

    /** Defines a point in the clipping plane */
    param::ParamSlot pointSlot;

    /** The plane-origin distance */
    param::ParamSlot distSlot;
};

} // namespace megamol::core::view
