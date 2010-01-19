/*
 * AnaglyphStereoDisplay.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES 1
#include "AnaglyphStereoDisplay.h"
#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */
#include <GL/gl.h>
#include "param/EnumParam.h"
#include "special/ClusterSignRenderer.h"
#include "view/CallCursorInput.h"
#include "view/CallRenderView.h"
#include "vislib/assert.h"
#include "vislib/Log.h"
#if defined(DEBUG) || defined(_DEBUG)
#include "vislib/Trace.h"
#endif
#include "vislib/mathfunctions.h"

using namespace megamol::core;


/*
 * special::AnaglyphStereoDisplay::AnaglyphStereoDisplay
 */
special::AnaglyphStereoDisplay::AnaglyphStereoDisplay(void)
        : AbstractStereoDisplay(),
        colorModeSlot("colourmode", "Specifies the colour separation mode.") {

    param::EnumParam *cmep = new param::EnumParam(0);
    cmep->SetTypePair(0, "Red-Cyan");
    cmep->SetTypePair(1, "Red-Green");
    cmep->SetTypePair(2, "Red-Blue");
    this->colorModeSlot << cmep;
    this->MakeSlotAvailable(&this->colorModeSlot);
}


/*
 * special::AnaglyphStereoDisplay::~AnaglyphStereoDisplay
 */
special::AnaglyphStereoDisplay::~AnaglyphStereoDisplay(void) {
    this->Release();
}


/*
 * special::AnaglyphStereoDisplay::Render
 */
void special::AnaglyphStereoDisplay::Render(void) {
    const GLboolean colMasks[][2][3] = {
         /*   left eye colour mask   */  /* right eye colour mask */
        {{GL_TRUE, GL_FALSE, GL_FALSE}, {GL_FALSE, GL_TRUE, GL_TRUE}},
        {{GL_TRUE, GL_FALSE, GL_FALSE}, {GL_FALSE, GL_TRUE, GL_FALSE}},
        {{GL_TRUE, GL_FALSE, GL_FALSE}, {GL_FALSE, GL_FALSE, GL_TRUE}}
    };

    view::CallRenderView *crv = this->getCallRenderView();
    if (crv != NULL) {
        int mode = this->colorModeSlot.Param<param::EnumParam>()->Value();
        ASSERT((mode >= 0) && (mode < 3));

        ::glViewport(0, 0, this->width(), this->height());

        ::glColorMask(colMasks[mode][0][0], colMasks[mode][0][1],
            colMasks[mode][0][2], GL_TRUE);
        crv->ResetAll();
        crv->SetProjection(vislib::graphics::CameraParameters::STEREO_OFF_AXIS,
            vislib::graphics::CameraParameters::LEFT_EYE);
        crv->SetTile(static_cast<float>(this->width()), static_cast<float>(this->height()),
            0.0f, 0.0f, static_cast<float>(this->width()), static_cast<float>(this->height()));
        crv->SetViewportSize(this->width(), this->height());
        (*crv)();

        ::glColorMask(colMasks[mode][1][0], colMasks[mode][1][1],
            colMasks[mode][1][2], GL_TRUE);
        crv->ResetAll();
        crv->SetProjection(vislib::graphics::CameraParameters::STEREO_OFF_AXIS,
            vislib::graphics::CameraParameters::RIGHT_EYE);
        crv->SetTile(static_cast<float>(this->width()), static_cast<float>(this->height()),
            0.0f, 0.0f, static_cast<float>(this->width()), static_cast<float>(this->height()));
        crv->SetViewportSize(this->width(), this->height());
        (*crv)();

        ::glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    } else {
        ::glViewport(0, 0, this->width(), this->height());
        special::ClusterSignRenderer::RenderBroken(
            this->width(), this->height());

    }

}


/*
 * special::AnaglyphStereoDisplay::create
 */
bool special::AnaglyphStereoDisplay::create(void) {
    // intentionally empty
    return AbstractStereoDisplay::create();
}


/*
 * special::AnaglyphStereoDisplay::release
 */
void special::AnaglyphStereoDisplay::release(void) {
    // intentionally empty
}
