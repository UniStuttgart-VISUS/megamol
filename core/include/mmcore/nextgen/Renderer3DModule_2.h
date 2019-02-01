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
#include "mmcore/view/RendererModule.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/view/MouseFlags.h"
#include "vislib/graphics/graphicstypes.h"
#include "mmcore/nextgen/CallRender3D_2.h"

namespace megamol {
namespace core {
namespace nextgen {

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
     * The chained get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times). This version of the method calls the respective method of alle chained renderers
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtentsChain(Call& call);

    /**
     * The callback that triggers the rendering of all chained render modules
     * as well as the own rendering
     *
     * @param call The calling call.
     *
     * @return True on success, false otherwise
     */
    virtual bool RenderChain(Call& call);

    // TODO events

    /**
     * The mouse event callback called when the mouse moved or a mouse
     * button changes it's state. This version of the method calls the respective method of alle chained renderers
     *
     * @param x The x coordinate of the mouse in world space
     * @param y The y coordinate of the mouse in world space
     * @param flags The mouse flags
     *
     * @return 'true' if the mouse event was consumed by the renderer and
     *         must be ignored by the view or subsequent renderer modules.
     */
    virtual bool MouseEventChain(float x, float y, view::MouseFlags flags);

private:
    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtentsCallback(CallRender3D_2& call) { return this->GetExtentsChain(call); }

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool RenderCallback(CallRender3D_2& call) { return this->RenderChain(call); }

    /** Slot for the daisy-chained renderer */
    CallerSlot chainRenderSlot;
};

} // namespace nextgen
} /* end namespace core */
} /* end namespace megamol */

#endif /** MEGAMOLCORE_Renderer3DModule_2_H_INCLUDED */
