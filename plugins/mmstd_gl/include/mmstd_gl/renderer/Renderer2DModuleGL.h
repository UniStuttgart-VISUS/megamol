/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "CallRender2DGL.h"
#include "mmcore/view/RendererModule.h"
#include "mmstd_gl/ModuleGL.h"

namespace megamol::mmstd_gl {

/**
 * Base class of rendering graph 2D renderer modules.
 */
class Renderer2DModuleGL : public core::view::RendererModule<CallRender2DGL, ModuleGL> {
public:
    /** Ctor. */
    Renderer2DModuleGL() : core::view::RendererModule<CallRender2DGL, ModuleGL>() {
        this->MakeSlotAvailable(&this->renderSlot);
    }

    /** Dtor. */
    ~Renderer2DModuleGL() override = default;

private:
    /**
     * The callback that triggers the own rendering and in theory would trigger
     * the rendering of all chained render modules
     *
     * @param call The calling call.
     *
     * @return True on success, false otherwise
     */
    bool RenderChain(CallRender2DGL& call) final;
};

inline bool Renderer2DModuleGL::RenderChain(CallRender2DGL& call) {

    // INSERT CHAINING HERE (IF EVER NEEDED)

    // bind fbo and set viewport before rendering your own stuff
    auto fbo = call.GetFramebuffer();
    fbo->bind();
    glViewport(0, 0, fbo->getWidth(), fbo->getHeight());

    // render our own stuff
    this->Render(call);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return true;
}

} // namespace megamol::mmstd_gl
