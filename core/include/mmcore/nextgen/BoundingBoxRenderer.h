/*
 * BoundingBoxRenderer.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_BOUNDINGBOXRENDERER_H_INCLUDED
#define MEGAMOLCORE_BOUNDINGBOXRENDERER_H_INCLUDED

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/nextgen/Renderer3DModule_2.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace core {
namespace nextgen {

/**
 * Renderer responsible for the rendering of the currently active bounding box as well as the view cube etc.
 */
class MEGAMOLCORE_API BoundingBoxRenderer : public Renderer3DModule_2 {
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
    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool GetExtents(core::nextgen::CallRender3D_2& call);

    /**
     * The Open GL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(core::nextgen::CallRender3D_2& call);

    /**
     * Render function for the bounding box
     *
     * @param call The incoming render call
     * @return True on success, false otherwise.
     */
    bool RenderBoundingBox(core::nextgen::CallRender3D_2& call);

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

    /** Parameter that enables or disables the view cube rendering */
    param::ParamSlot enableViewCubeSlot;

    /** Handle of the vertex buffer object */
    GLuint vbo;

    /** Handle of the vertex array to be rendered */
    GLuint va;
};
} // namespace nextgen
} // namespace core
} // namespace megamol

#endif /* MEGAMOLCORE_BOUNDINGBOXRENDERER_H_INCLUDED */
