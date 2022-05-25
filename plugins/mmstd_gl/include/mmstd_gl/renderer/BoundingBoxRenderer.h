/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>

#include <glowl/glowl.h>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/RendererModule.h"
#include "mmstd_gl/ModuleGL.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"

namespace megamol::mmstd_gl {

/**
 * Renderer responsible for the rendering of the currently active bounding box as well as the view cube etc.
 * This is a special renderer without the typical structure of other renderers, since it does not inherit from
 * mmstd_gl::Renderer3DModuleGL.
 */
class BoundingBoxRenderer : public core::view::RendererModule<CallRender3DGL, ModuleGL> {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "BoundingBoxRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Renders the incoming bounding box as well as the view cube etc.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    BoundingBoxRenderer();

    /** Dtor. */
    ~BoundingBoxRenderer() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

private:
    /*
     * Copies the incoming call to the outgoing one to pass the extents
     *
     * @param call The call containing all relevant parameters
     * @return True on success, false otherwise
     */
    bool GetExtents(CallRender3DGL& call) override;

    /*
     * Renders the bounding box and the viewcube on top of the other rendered things
     *
     * @param call The call containing the camera and other parameters
     * @return True on success, false otherwise
     */
    bool Render(CallRender3DGL& call) final;

    /**
     * Render function for the bounding box front
     *
     * @param call The used mvp matrix
     * @param bb The bounding box to render
     * @param smoothLines Determines whether the lines of the box should get smoothed. default = true
     * @return True on success, false otherwise.
     */
    bool RenderBoundingBoxFront(const glm::mat4& mvp, const core::BoundingBoxes_2& bb, bool smoothLines = true);

    /**
     * Render function for the bounding box back
     *
     * @param mvp Model, view and projection matrices combined
     * @param bb The bounding box to render
     * @param smoothLines Determines whether the lines of the box should get smoothed. default = true
     * @return True on success, false otherwise.
     */
    bool RenderBoundingBoxBack(const glm::mat4& mvp, const core::BoundingBoxes_2& bb, bool smoothLines = true);

    /**
     * Render function for the view cube
     *
     * @param call The call containing the camera and other parameters
     * @return True on success, false otherwise.
     */
    bool RenderViewCube(CallRender3DGL& call);

    /** Parameter that enables or disables the bounding box rendering */
    core::param::ParamSlot enableBoundingBoxSlot;

    /** Parameter storing the desired color of the bounding box */
    core::param::ParamSlot boundingBoxColorSlot;

    /** Parameter enabling or disabling the smoothing of lines */
    core::param::ParamSlot smoothLineSlot;

    /** Parameter that enables or disables the view cube rendering */
    core::param::ParamSlot enableViewCubeSlot;

    /** Parameter for setting the position of the view cube */
    core::param::ParamSlot viewCubePosSlot;

    /** Parameter for setting the view cube size */
    core::param::ParamSlot viewCubeSizeSlot;

    /** Handle of the vertex buffer object */
    GLuint vbo;

    /** Handle of the index buffer object */
    GLuint ibo;

    /** Handle of the vertex array to be rendered */
    GLuint va;

    /** Shader program for lines */
    std::unique_ptr<glowl::GLSLProgram> lineShader;

    /** Shader program for a cube */
    std::unique_ptr<glowl::GLSLProgram> cubeShader;

    /** Bounding Boxes */
    megamol::core::BoundingBoxes_2 boundingBoxes;
};
} // namespace megamol::mmstd_gl
