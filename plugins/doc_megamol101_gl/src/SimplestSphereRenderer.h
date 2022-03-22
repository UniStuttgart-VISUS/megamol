/**
 * MegaMol
 * Copyright (c) 2018, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOL_MEGAMOL101_SIMPLESTSPHERERENDERER_H
#define MEGAMOL_MEGAMOL101_SIMPLESTSPHERERENDERER_H

#include <memory>

#include <glowl/glowl.h>

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"

namespace megamol::megamol101_gl {

/**
 * Renders incoming spheres to the screen, either using GL_POINTS or more
 * sophisticated shaders.
 */
class SimplestSphereRenderer : public core_gl::view::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * TUTORIAL: Mandatory method for every module or call that states the name of the class.
     * This name should be unique.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "SimplestSphereRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * TUTORIAL: Mandatory method for every module or call that returns a description.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Renders a set of incoming spheres";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * TUTORIAL: Mandatory method for every module.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Constructor. */
    SimplestSphereRenderer();

    /** Destructor. */
    ~SimplestSphereRenderer() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * TUTORIAL: The overwritten version of this method gets called right after an object of this class has been
     * instantiated. Shader compilation should be done inside this method.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     *
     * TUTORIAL: The overwritten version of this method gets called on destruction of this object.
     * Necessary cleanup should be done inside this method.
     */
    void release() override;

private:
    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * TUTORIAL: This method computes the extents of the rendered data, namely the bounding box and other relevant
     *  values, and writes them into the calling call.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    bool GetExtents(core_gl::view::CallRender3DGL& call) override;

    /**
     * The Open GL Render callback.
     *
     * TUTORIAL: Mandatory method for each renderer. It performs the main OpenGL rendering.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    bool Render(core_gl::view::CallRender3DGL& call) override;

    /** The input data slot. */
    core::CallerSlot sphereDataSlot;

    /** The vertex buffer object for the rendered vertices. */
    GLuint vbo;

    /** The vertex array for the rendered vertices. */
    GLuint va;

    /** The data hash of the most recent rendered data */
    std::size_t lastDataHash;

    /** The simple shader for the drawing of GL_POINTS */
    std::unique_ptr<glowl::GLSLProgram> simpleShader;

    /** The pretty shader that draws spheres*/
    std::unique_ptr<glowl::GLSLProgram> sphereShader;

    /** Slot for the scaling factor of the pointsize*/
    core::param::ParamSlot sizeScalingSlot;

    /** Slot for the switch between GL_POINTS and spheres */
    core::param::ParamSlot sphereModeSlot;
};

} // namespace megamol::megamol101_gl

#endif // MEGAMOL_MEGAMOL101_SIMPLESTSPHERERENDERER_H
