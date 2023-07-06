/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmstd/renderer/CallRender3D.h"
#include "mmstd/renderer/RendererModule.h"

namespace megamol::core::view {

/**
 * New and improved base class of rendering graph 3D renderer modules.
 */
class Renderer3DModule : public view::RendererModule<CallRender3D, Module> {
public:
    /** Ctor. */
    Renderer3DModule();

    /** Dtor. */
    ~Renderer3DModule() override;

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
    bool GetExtents(CallRender3D& call) override = 0;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(CallRender3D& call) override = 0;

    /**
     * Method that gets called before the rendering is started for all changed modules
     *
     * @param call The rendering call that contains the camera
     */
    virtual void PreRender(CallRender3D& call);

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
    bool GetExtentsChain(CallRender3D& call) final;

    /**
     * The callback that triggers the rendering of all chained render modules
     * as well as the own rendering
     *
     * @param call The calling call.
     *
     * @return True on success, false otherwise
     */
    bool RenderChain(CallRender3D& call) final;

    // TODO events
};

} // namespace megamol::core::view
