/*
 * QuartzPlaneRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <memory>

#include <glowl/glowl.h>

#include "AbstractMultiShaderQuartzRenderer.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/renderer/CallRender2DGL.h"
#include "mmstd_gl/renderer/Renderer2DModuleGL.h"


namespace megamol::demos_gl {

/**
 * QuartzPlaneRenderer
 */
class QuartzPlaneRenderer : public mmstd_gl::Renderer2DModuleGL, public AbstractMultiShaderQuartzRenderer {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "QuartzPlaneRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Module rendering gridded quartz particles onto a clipping plane";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return AbstractMultiShaderQuartzRenderer::IsAvailable();
    }

    /**
     * Ctor
     */
    QuartzPlaneRenderer();

    /**
     * Dtor
     */
    ~QuartzPlaneRenderer() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender2DGL& call) override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender2DGL& call) override;

    /**
     * Creates a raycasting shader for the specified crystalite
     *
     * @param c The crystalite
     *
     * @return The shader
     */
    std::shared_ptr<glowl::GLSLProgram> makeShader(const CrystalDataCall::Crystal& c) override;

private:
    /** Use clipping plane or grain colour for grains */
    core::param::ParamSlot useClipColSlot;
};

} // namespace megamol::demos_gl
