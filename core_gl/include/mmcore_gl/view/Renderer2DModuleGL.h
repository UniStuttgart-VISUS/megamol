/*
 * Renderer2DModule.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_RENDERER2DMODULE_H_INCLUDED
#define MEGAMOLCORE_RENDERER2DMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallRender2DGL.h"
#include "mmcore/view/RendererModule.h"
#include "mmcore_gl/ModuleGL.h"


namespace megamol {
namespace core_gl {
namespace view {

/**
 * Base class of rendering graph 2D renderer modules.
 */
class Renderer2DModuleGL : public core::view::RendererModule<CallRender2DGL, ModuleGL> {
public:
    /** Ctor. */
    Renderer2DModuleGL();

    /** Dtor. */
    virtual ~Renderer2DModuleGL();

protected:
    /**
     * Method that gets called before the rendering is started for all changed modules
     *
     * @param call The rendering call that contains the camera
     */
    virtual void PreRender(CallRender2DGL& call);

private:
    /**
     * The chained get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times). This version of the method calls the respective method of all chained renderers
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtentsChain(CallRender2DGL& call) override final;

    /**
     * The callback that triggers the own rendering and in theory would trigger
     * the rendering of all chained render modules
     *
     * @param call The calling call.
     *
     * @return True on success, false otherwise
     */
    virtual bool RenderChain(CallRender2DGL& call) override final;
};

} /* end namespace view */
} // namespace core_gl
} /* end namespace megamol */

#endif /* MEGAMOLCORE_RENDERERMODULE_H_INCLUDED */
