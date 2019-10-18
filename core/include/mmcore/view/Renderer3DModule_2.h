/*
 * Renderer3DModule_2.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_Renderer3DModule_2_H_INCLUDED
#define MEGAMOLCORE_Renderer3DModule_2_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/MouseFlags.h"
#include "mmcore/view/RendererModule.h"
#include "mmcore/view/light/CallLight.h"
#include "vislib/graphics/graphicstypes.h"

namespace megamol {
namespace core {
namespace view {

/**
 * New and improved base class of rendering graph 3D renderer modules.
 */
class MEGAMOLCORE_API Renderer3DModule_2 : public view::RendererModule<CallRender3D_2> {
public:
    /** Ctor. */
    Renderer3DModule_2(void);

    /** Dtor. */
    virtual ~Renderer3DModule_2(void);

protected:
    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(CallRender3D_2& call) = 0;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(CallRender3D_2& call) = 0;

    /**
     * Receives the current lights from the light call and writes them to the lightMap
     *
     * @return True if any light has changed, false otherwise.
     */
    bool GetLights(void);

	/**
	 * Method that gets called before the rendering is started for all changed modules
	 *
	 * @param call The rendering call that contains the camera
	 */
	virtual void PreRender(CallRender3D_2& call);

    /** map to store the called lights */
    core::view::light::LightMap lightMap;

private:
    /**
     * The chained get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times). This version of the method calls the respective method of alle chained renderers
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtentsChain(CallRender3D_2& call) override final;

    /**
     * The callback that triggers the rendering of all chained render modules
     * as well as the own rendering
     *
     * @param call The calling call.
     *
     * @return True on success, false otherwise
     */
    virtual bool RenderChain(CallRender3D_2& call) override final;

    // TODO events

    /** Slot to retrieve the light information */
    CallerSlot lightSlot;
};

} // namespace view
} /* end namespace core */
} /* end namespace megamol */

#endif /** MEGAMOLCORE_Renderer3DModule_2_H_INCLUDED */
