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
    Renderer2DModuleGL() : core::view::RendererModule<CallRender2DGL, ModuleGL>() {
        this->MakeSlotAvailable(&this->renderSlot);
    }

    /** Dtor. */
    virtual ~Renderer2DModuleGL(void) = default;

private:
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

} /* end namespace view */
} // namespace core_gl
} /* end namespace megamol */

#endif /* MEGAMOLCORE_RENDERERMODULE_H_INCLUDED */
