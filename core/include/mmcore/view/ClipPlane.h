/*
 * ClipPlane.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLIPPLANE_H_INCLUDED
#define MEGAMOLCORE_CLIPPLANE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/math/Plane.h"


namespace megamol {
namespace core {
namespace view {


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
    static const char* ClassName(void) {
        return "ClipPlane";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Module defining a clipping plane";
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
    ClipPlane(void);

    /** Dtor. */
    virtual ~ClipPlane(void);

private:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

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


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLIPPLANE_H_INCLUDED */
