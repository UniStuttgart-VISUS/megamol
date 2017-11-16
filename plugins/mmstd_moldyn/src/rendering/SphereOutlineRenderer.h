/*
 * SphereOutlineRenderer.h
 *
 * Copyright (C) 2009-2017 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MMSTD_MOLDYN_SPHEREOUTLINERENDERER_H_INCLUDED
#define MMSTD_MOLDYN_SPHEREOUTLINERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace stdplugin {
namespace moldyn {
namespace rendering {

/**
 * Renderer for simple sphere glyphs
 */
class SphereOutlineRenderer : public core::view::Renderer3DModule {
public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "SphereOutlineRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Renderer for outlines of sphere glyphs";
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
    SphereOutlineRenderer(void);

    /** Dtor. */
    virtual ~SphereOutlineRenderer(void);

protected:

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * The get capabilities callback. The module should set the members
     * of 'call' to tell the caller its capabilities.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetCapabilities(core::Call& call);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(core::Call& call);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(core::Call& call);

private:

    /** The call for data */
    core::CallerSlot getDataSlot;

    /** The base colour for the sphere outline */
    core::param::ParamSlot colourSlot;

    /** The representation type */
    core::param::ParamSlot repSlot;

    /** The number of line segments to construct the circle/sphere */
    core::param::ParamSlot circleSegSlot;

    /** The (half) number of additional outlines */
    core::param::ParamSlot multiOutlineCntSlot;

    /** The distance of the additional outlines as angles in radians */
    core::param::ParamSlot multiOutLineDistSlot;

    /** The sphere quadric */
    void *sphereQuadric;

};

} /* end namespace rendering */
} /* end namespace moldyn */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MMSTD_MOLDYN_SPHEREOUTLINERENDERER_H_INCLUDED */
