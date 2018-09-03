/*
 * SombreroMeshRenderer.h
 * Copyright (C) 2006-2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MMSOMBREROSPLUGIN_SOMBREROMESHRENDERER_H_INCLUDED
#define MMSOMBREROSPLUGIN_SOMBREROMESHRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "vislib/math/Cuboid.h"
#include "vislib/memutils.h"


namespace megamol {
namespace sombreros {

/**
 * Renderer for tri-mesh data
 */
class SombreroMeshRenderer : public core::view::Renderer3DModule {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "SombreroMeshRenderer"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Renderer for sombrero tri-mesh data"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    SombreroMeshRenderer(void);

    /** Dtor. */
    virtual ~SombreroMeshRenderer(void);

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
    /** The slot to fetch the data */
    core::CallerSlot getDataSlot;

    /** The slot to fetch the volume data */
    core::CallerSlot getVolDataSlot;

    /** Flag whether or not to show vertices */
    core::param::ParamSlot showVertices;

    /** Flag whether or not use lighting for the surface */
    core::param::ParamSlot lighting;

    /** The rendering style for the front surface */
    core::param::ParamSlot surFrontStyle;

    /** The rendering style for the back surface */
    core::param::ParamSlot surBackStyle;

    /** The Triangle winding rule */
    core::param::ParamSlot windRule;

    /** The Triangle color */
    core::param::ParamSlot colorSlot;

    /** Slot to activate scaling */
    core::param::ParamSlot doScaleSlot;

    /** Slot to activate the display of the sweatband */
    core::param::ParamSlot showSweatBandSlot;
};


} /* end namespace sombreros */
} /* end namespace megamol */

#endif /* MMSOMBREROSPLUGIN_SOMBREROMESHRENDERER_H_INCLUDED */
