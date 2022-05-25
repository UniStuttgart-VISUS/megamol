/**
 * MegaMol
 * Copyright (c) 2010, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/factories/ModuleAutoDescription.h"
#include "mmcore_gl/view/Renderer2DModuleGL.h"

namespace megamol::core_gl::view::special {

/**
 * A simple 2d renderer which just creates a circle
 */
class ChronoGraph : public core_gl::view::Renderer2DModuleGL {
public:
    /**
     * The class name for the factory
     *
     * @return The class name
     */
    static const char* ClassName() {
        return "ChronoGraph";
    }

    /**
     * A human-readable description string for the module
     *
     * @return The description string
     */
    static const char* Description() {
        return "ChronoGraph renderer displaying the core instance time";
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
    ChronoGraph();

    /**
     * dtor
     */
    ~ChronoGraph() override;

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

private:
    /**
     * Renders the info grid into the given rectangle
     *
     * @param time The time code to show
     * @param x The x coordinate
     * @param y The y coordinate
     * @param w The width
     * @param h The height
     */
    void renderInfoGrid(float time, float x, float y, float w, float h);

    /**
     * Renders the info circle into the given rectangle
     *
     * @param time The time code to show
     * @param x The x coordinate
     * @param y The y coordinate
     * @param w The width
     * @param h The height
     */
    void renderInfoCircle(float time, float x, float y, float w, float h);
};

} // namespace megamol::core_gl::view::special
