/**
 * MegaMol
 * Copyright (c) 2018, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmstd/renderer/RendererModule.h"
#include "mmstd_gl/ModuleGL.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"

namespace megamol::mmstd_gl {

/**
 * New and improved base class of rendering graph 3D renderer modules.
 */
class Renderer3DModuleGL : public core::view::RendererModule<CallRender3DGL, ModuleGL> {
public:
    /** Ctor. */
    Renderer3DModuleGL();

    /** Dtor. */
    ~Renderer3DModuleGL() override;

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
    bool GetExtents(CallRender3DGL& call) override = 0;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(CallRender3DGL& call) override = 0;

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
    bool GetExtentsChain(CallRender3DGL& call) final;

    /**
     * The callback that triggers the rendering of all chained render modules
     * as well as the own rendering
     *
     * @param call The calling call.
     *
     * @return True on success, false otherwise
     */
    bool RenderChain(CallRender3DGL& call) final;

    // TODO events
};

} // namespace megamol::mmstd_gl
