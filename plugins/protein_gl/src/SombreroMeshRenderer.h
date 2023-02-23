/*
 * SombreroMeshRenderer.h
 * Copyright (C) 2006-2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/utility/SDFFont.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "vislib/math/Cuboid.h"
#include "vislib/memutils.h"

#include <set>

#include "mmstd_gl/renderer/CallRender3DGL.h"

namespace megamol::protein_gl {

/**
 * Renderer for tri-mesh data
 */
class SombreroMeshRenderer : public mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "SombreroMeshRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Renderer for sombrero tri-mesh data";
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
    SombreroMeshRenderer();

    /** Dtor. */
    ~SombreroMeshRenderer() override;

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
    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

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
    bool Render(mmstd_gl::CallRender3DGL& call) override;

#if 0
    /**
     * Callback for mouse events (move, press, and release)
     *
     * @param x The x coordinate of the mouse in world space
     * @param y The y coordinate of the mouse in world space
     * @param flags The mouse flags
     */
    virtual bool MouseEvent(float x, float y, megamol::core::view::MouseFlags flags);
#endif

private:
    struct lastCamState_t {
        vislib::math::Vector<float, 3> camPos;
        vislib::math::Vector<float, 3> camDir;
        vislib::math::Vector<float, 3> camUp;
        vislib::math::Vector<float, 3> camRight;
        float zNear;
        float zFar;
        float fovx;
        float fovy;
        float aspect;
        int width;
        int height;
    } lastCamState;

    inline std::string trunc(const float val, const unsigned int numDigits = 2) {
        std::string t = std::to_string(static_cast<int>(val));
        std::string output = std::to_string(val).substr(0, numDigits + t.length() + 1);
        if (output.find('.') == std::string::npos || output.back() == '.') {
            output.pop_back();
        }
        return output;
    }

    /**
     * Returns the direction a pixel lies in the image plane relative to the camera
     * This vector can then be used for picking, for example.
     *
     * @param x The mouse coordinate in x-direction.
     * @param y The mouse coordinate in y-direction.
     */
    vislib::math::Vector<float, 3> getPixelDirection(float x, float y);

    /**
     * Performs an intersection test of a ray with a given triangle
     */
    bool rayTriIntersect(const vislib::math::Vector<float, 3>& pos, const vislib::math::Vector<float, 3>& dir,
        const vislib::math::Vector<float, 3>& p1, const vislib::math::Vector<float, 3>& p2,
        const vislib::math::Vector<float, 3>& p3, float& intersectDist);

    /**
     * Overrides the selected colors
     */
    void overrideColors(const int meshIdx, const vislib::math::Vector<float, 3>& color);

    /** The slot to fetch the data */
    core::CallerSlot getDataSlot;

    /** The slot to fetch the volume data */
    core::CallerSlot getVolDataSlot;

    /** Slot connecting this module to the flag storage */
    core::CallerSlot getFlagDataSlot;

    /** Flag whether or not to show vertices */
    core::param::ParamSlot showVertices;

    /** Flag whether or not use lighting for the surface */
    core::param::ParamSlot lighting;

    /** The rendering style for the front surface */
    core::param::ParamSlot surFrontStyle;

    /** The rendering style for the back surface */
    core::param::ParamSlot surBackStyle;

    /** The Triangle winding rule */
    core::param::ParamSlot windRule;

    /** The Triangle color */
    core::param::ParamSlot colorSlot;

    /** The color for the brushing */
    core::param::ParamSlot brushColorSlot;

    /** The color for the inner line */
    core::param::ParamSlot innerColorSlot;

    /** The color for the sweatband line */
    core::param::ParamSlot borderColorSlot;

    /** The color for the outer line */
    core::param::ParamSlot outerColorSlot;

    /** Thec color for the font */
    core::param::ParamSlot fontColorSlot;

    /** Slot to activate the display of the sweatband */
    core::param::ParamSlot showSweatBandSlot;

    /** Slot controlling the annotation of radii */
    core::param::ParamSlot showRadiiSlot;

    /** The font used to render text */
    core::utility::SDFFont theFont;

    /** The last time value asked by the render routine */
    float lastTime;

    std::vector<std::vector<vislib::math::Vector<float, 3>>> vertexPositions;

    std::vector<std::vector<vislib::math::Vector<unsigned int, 3>>> triangles;

    std::vector<std::vector<unsigned int>> indexAttrib;

    std::set<unsigned int> flagSet;

    uint32_t flagVersion;

    std::vector<std::vector<float>> newColors;

    bool changedFlags;

    size_t lastDataHash;
};


} // namespace megamol::protein_gl
