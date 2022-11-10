//
// VariantMatchRenderer.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jun 15, 2013
//     Author: scharnkn
//

#pragma once

#include <memory>

#include <glowl/glowl.h>

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/renderer/Renderer2DModuleGL.h"
#include "vislib_gl/graphics/gl/OutlineFont.h"
#include "vislib_gl/graphics/gl/Verdana.inc"


namespace megamol {
namespace protein_gl {


class VariantMatchRenderer : public megamol::mmstd_gl::Renderer2DModuleGL {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "VariantMatchRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Offers matrix-like rendering of variant matchings";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Dtor */
    VariantMatchRenderer(void);

    /** Ctor */
    ~VariantMatchRenderer(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(mmstd_gl::CallRender2DGL& call);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * The Open GL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(mmstd_gl::CallRender2DGL& call);

    /**
     * Update all parameter slots
     */
    void updateParams();

private:
    /**
     * Draw the color map using the current min/max values.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool drawColorMap();

    /// The data caller slot
    core::CallerSlot dataCallerSlot;


    /* Parameter slots */

    /// Minimum value for color
    core::param::ParamSlot minColSlot;
    float minCol;

    /// Maximum value for color
    core::param::ParamSlot maxColSlot;
    float maxCol;

    /// The texturing shader
    std::unique_ptr<glowl::GLSLProgram> matrixTexShader;

    /// The texturing shader
    std::unique_ptr<glowl::GLSLProgram> colorMapShader;

    /// The matix texture handle
    GLuint matrixTex;

    /// The outline font
    vislib_gl::graphics::gl::OutlineFont thefont;

    float fontSize;
};

} // namespace protein_gl
} // end namespace megamol
