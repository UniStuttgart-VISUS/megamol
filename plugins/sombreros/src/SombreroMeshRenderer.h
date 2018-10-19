/*
 * SombreroMeshRenderer.h
 * Copyright (C) 2006-2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MMSOMBREROSPLUGIN_SOMBREROMESHRENDERER_H_INCLUDED
#define MMSOMBREROSPLUGIN_SOMBREROMESHRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "vislib/math/Cuboid.h"
#include "vislib/memutils.h"

#include <set>


namespace megamol {
namespace sombreros {

/**
 * Renderer for tri-mesh data
 */
class SombreroMeshRenderer : public core::view::Renderer3DModule {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "SombreroMeshRenderer"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Renderer for sombrero tri-mesh data"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /** Ctor. */
    SombreroMeshRenderer(void);

    /** Dtor. */
    virtual ~SombreroMeshRenderer(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * The get capabilities callback. The module should set the members
     * of 'call' to tell the caller its capabilities.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetCapabilities(core::Call& call);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(core::Call& call);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(core::Call& call);

    /**
     * Callback for mouse events (move, press, and release)
     *
     * @param x The x coordinate of the mouse in world space
     * @param y The y coordinate of the mouse in world space
     * @param flags The mouse flags
     */
    virtual bool MouseEvent(float x, float y, megamol::core::view::MouseFlags flags);

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

    /** Slot to activate scaling */
    core::param::ParamSlot doScaleSlot;

    /** Slot to activate the display of the sweatband */
    core::param::ParamSlot showSweatBandSlot;

    /** The last time value asked by the render routine */
    float lastTime;

    std::vector<std::vector<vislib::math::Vector<float, 3>>> vertexPositions;

    std::vector<std::vector<vislib::math::Vector<unsigned int, 3>>> triangles;

    std::vector<std::vector<unsigned int>> indexAttrib;

    std::set<unsigned int> flagSet;

    std::vector<std::vector<float>> newColors;

    bool changedFlags;

    size_t lastDataHash;
};


} /* end namespace sombreros */
} /* end namespace megamol */

#endif /* MMSOMBREROSPLUGIN_SOMBREROMESHRENDERER_H_INCLUDED */
