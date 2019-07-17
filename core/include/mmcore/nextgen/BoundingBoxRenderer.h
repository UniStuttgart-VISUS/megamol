/*
 * BoundingBoxRenderer.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_BOUNDINGBOXRENDERER_H_INCLUDED
#define MEGAMOLCORE_BOUNDINGBOXRENDERER_H_INCLUDED

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/nextgen/CallRender3D_2.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/RendererModule.h"
#include "vislib/graphics/gl/GLSLShader.h"

namespace megamol {
namespace core {
namespace nextgen {

/**
 * Renderer responsible for the rendering of the currently active bounding box as well as the view cube etc.
 * This is a special renderer without the typical structure of other renderers, since it does not inherit from
 * Renderer3DModule_2.
 */
class MEGAMOLCORE_API BoundingBoxRenderer : public view::RendererModule<CallRender3D_2> {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "BoundingBoxRenderer"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Renders the incoming bounding box as well as the view cube etc."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    BoundingBoxRenderer(void);

    /** Dtor. */
    virtual ~BoundingBoxRenderer(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

private:
    /*
     * Copies the incoming call to the outgoing one to pass the extents
     *
     * @param call The call containing all relevant parameters
     * @return True on success, false otherwise
     */
    virtual bool GetExtents(core::nextgen::CallRender3D_2& call) override;

    /*
     * Renders the bounding box and the viewcube on top of the other rendered things
     *
     * @param call The call containing the camera and other parameters
     * @return True on success, false otherwise
     */
    virtual bool Render(core::nextgen::CallRender3D_2& call) override final;

    /**
     * Render function for the bounding box front
     *
     * @param call The used mvp matrix
     * @param bb The bounding box to render
     * @param smoothLines Determines whether the lines of the box should get smoothed. default = true
     * @return True on success, false otherwise.
     */
    bool RenderBoundingBoxFront(const glm::mat4& mvp, const BoundingBoxes_2& bb, bool smoothLines = true);

    /**
     * Render function for the bounding box back
     *
     * @param call The incoming render call
     * @param bb The bounding box to render
     * @param smoothLines Determines whether the lines of the box should get smoothed. default = true
     * @return True on success, false otherwise.
     */
    bool RenderBoundingBoxBack(const glm::mat4& mvp, const BoundingBoxes_2& bb, bool smoothLines = true);

    /**
     * Render function for the view cube
     *
     * @param call The incoming render call
     * @return True on success, false otherwise.
     */
    bool RenderViewCube(core::nextgen::CallRender3D_2& call);

    /** Parameter that enables or disables the bounding box rendering */
    param::ParamSlot enableBoundingBoxSlot;

    /** Parameter storing the desired color of the bounding box */
    param::ParamSlot boundingBoxColorSlot;

    /** Parameter enabling or disabling the smoothing of lines */
    param::ParamSlot smoothLineSlot;

    /** Parameter that enables or disables the view cube rendering */
    param::ParamSlot enableViewCubeSlot;

    /** Handle of the vertex buffer object */
    GLuint vbo;

    /** Handle of the index buffer object */
    GLuint ibo;

    /** Handle of the vertex array to be rendered */
    GLuint va;

    /** Shader program for lines */
    vislib::graphics::gl::GLSLShader lineShader;
};
} // namespace nextgen
} // namespace core
} // namespace megamol

#endif /* MEGAMOLCORE_BOUNDINGBOXRENDERER_H_INCLUDED */
