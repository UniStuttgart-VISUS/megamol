/*
 * core_gl::view::Renderer3DModuleGL.h
 *
 * Copyright (C) 2018, 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_Renderer3DModuleGL_H_INCLUDED
#define MEGAMOLCORE_Renderer3DModuleGL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/RendererModule.h"
#include "mmcore_gl/ModuleGL.h"
#include "mmcore_gl/view/CallRender3DGL.h"

namespace megamol {
namespace core_gl {
namespace view {

/**
 * New and improved base class of rendering graph 3D renderer modules.
 */
class Renderer3DModuleGL : public core::view::RendererModule<CallRender3DGL, ModuleGL> {
public:
    /** Ctor. */
    Renderer3DModuleGL(void);

    /** Dtor. */
    virtual ~Renderer3DModuleGL(void);

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
    virtual bool GetExtents(CallRender3DGL& call) = 0;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(CallRender3DGL& call) = 0;

    /**
     * Method that gets called before the rendering is started for all changed modules
     *
     * @param call The rendering call that contains the camera
     */
    virtual void PreRender(CallRender3DGL& call);

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
    virtual bool GetExtentsChain(CallRender3DGL& call) override final;

    /**
     * The callback that triggers the rendering of all chained render modules
     * as well as the own rendering
     *
     * @param call The calling call.
     *
     * @return True on success, false otherwise
     */
    virtual bool RenderChain(CallRender3DGL& call) override final;

    // TODO events
};

} // namespace view
} // namespace core_gl
} /* end namespace megamol */

#endif /** MEGAMOLCORE_Renderer3DModuleGL_H_INCLUDED */
