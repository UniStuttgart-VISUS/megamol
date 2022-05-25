/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/factories/ModuleAutoDescription.h"
#include "mmcore/view/MouseFlags.h"
#include "mmcore_gl/view/Renderer2DModuleGL.h"

namespace megamol::core_gl::view::special {

/**
 * A simple 2d renderer which just creates a circle
 */
class DemoRenderer2D : public core_gl::view::Renderer2DModuleGL {
public:
    /**
     * The class name for the factory
     *
     * @return The class name
     */
    static const char* ClassName() {
        return "DemoRenderer2D";
    }

    /**
     * A human-readable description string for the module
     *
     * @return The description string
     */
    static const char* Description() {
        return "Demo 2D-Renderer";
    }

    /**
     * Test if the module can be instanziated
     *
     * @return 'true'
     */
    static bool IsAvailable() {
        return true;
    }

    /**
     * Disallow usage in quickstarts
     *
     * @return false
     */
    static bool SupportQuickstart() {
        return false;
    }

    /**
     * ctor
     */
    DemoRenderer2D();

    /**
     * dtor
     */
    ~DemoRenderer2D() override;

protected:
    /**
     * Initializes the module directly after instanziation
     *
     * @return 'true' on success
     */
    bool create() override;

    /**
     * Sets the extents (animation and bounding box) into the call object
     *
     * @param call The incoming call
     *
     * @return 'true' on success
     */
    bool GetExtents(core_gl::view::CallRender2DGL& call) override;

    /**
     * Renders the scene
     *
     * @param call The incoming call
     *
     * @return 'true' on success
     */
    bool Render(core_gl::view::CallRender2DGL& call) override;

    /**
     * Releases all resources of the module
     */
    void release() override;

    /**
     * Callback for mouse events (move, press, and release)
     *
     * @param x The x coordinate of the mouse in world space
     * @param y The y coordinate of the mouse in world space
     * @param flags The mouse flags
     */
    bool MouseEvent(float x, float y, core::view::MouseFlags flags) override;

private:
    /** The mouse coordinate */
    float mx, my;

    /** The coordinates to draw the test line from */
    float fromx, fromy;

    /** The coordinates to draw the test line to */
    float tox, toy;

    /** Flag if the test line is being spanned */
    bool drag;
};

} // namespace megamol::core_gl::view::special
