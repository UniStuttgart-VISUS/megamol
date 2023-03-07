/*
 * TimeLineRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "RuntimeConfig.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/renderer/CallRender2DGL.h"
#include "mmstd_gl/renderer/Renderer2DModuleGL.h"

#include "cinematic/CallKeyframeKeeper.h"
#include "cinematic/Keyframe.h"
#include "cinematic_gl/CinematicUtils.h"


namespace megamol::cinematic_gl {

/**
 * Timeline rendering.
 */
class TimeLineRenderer : public mmstd_gl::Renderer2DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TimeLineRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Renders the timeline of keyframes";
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
    TimeLineRenderer();

    /** Dtor. */
    ~TimeLineRenderer() override;

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
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender2DGL& call) override;

    /**
     * The mouse button pressed/released callback.
     */
    bool OnMouseButton(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action,
        megamol::core::view::Modifiers mods) override;

    /**
     * The mouse movement callback.
     */
    bool OnMouseMove(double x, double y) override;

private:
    /**********************************************************************
     * variables
     **********************************************************************/

    struct AxisData {
        glm::vec2 startPos;
        glm::vec2 endPos;
        float length;
        float maxValue;
        float segmSize;  // the world space size of one segment
        float segmValue; // value of on segment on the ruler
        float scaleFactor;
        float scaleOffset;         // negative offset to keep position on the ruler during scaling in focus
        float scaleDelta;          // scaleOffset for new scalePos to get new scaleOffset for new scaling factor
        float valueFractionLength; // the scaled fraction of the axis length and the max value
        float rulerPos;
        std::string formatStr; // string with adapted floating point formatting
    };

    enum Axis : size_t { X = 0, Y = 1, COUNT = 2 };

    enum ActiveParam : size_t { SIMULATION_TIME };

    std::array<AxisData, Axis::COUNT> axes;
    CinematicUtils utils;
    GLuint texture_id;
    ActiveParam yAxisParam;
    cinematic::Keyframe dragDropKeyframe;
    bool dragDropActive;
    unsigned int axisDragDropMode;
    unsigned int axisScaleMode;
    float keyframeMarkSize;
    float rulerMarkHeight;
    glm::vec2 viewport;
    unsigned int fps;
    float mouseX;
    float mouseY;
    float lastMouseX;
    float lastMouseY;
    core::view::MouseButton mouseButton;
    core::view::MouseButtonAction mouseAction;
    float lineHeight;

    /**********************************************************************
     * functions
     **********************************************************************/

    bool recalcAxesData();

    void pushMarkerTexture(float pos_x, float pos_y, float size, glm::vec4 color);

    /**********************************************************************
     * callbacks
     **********************************************************************/

    core::CallerSlot keyframeKeeperSlot;

    /**********************************************************************
     * parameters
     **********************************************************************/

    megamol::core::param::ParamSlot moveRightFrameParam;
    megamol::core::param::ParamSlot moveLeftFrameParam;
    megamol::core::param::ParamSlot resetPanScaleParam;
};

} // namespace megamol::cinematic_gl
