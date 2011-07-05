/*
 * RelativeCursor3D.cpp
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "vislib/RelativeCursor3D.h"
#include "vislib/mathfunctions.h"
#include "vislib/AbstractCursor3DEvent.h"
#include <stdio.h>


/*
 * vislib::graphics::RelativeCursor3D::RelativeCursor3D
 */
vislib::graphics::RelativeCursor3D::RelativeCursor3D(void) : AbstractCursor(),
        translate(), rotate(), camPams(NULL) { 
    // intentionally empty
}


/*
 * vislib::graphics::RelativeCursor3D::RelativeCursor3D
 */
vislib::graphics::RelativeCursor3D::RelativeCursor3D(
        const RelativeCursor3D& rhs) : AbstractCursor(rhs),
        translate(rhs.translate), rotate(rhs.rotate), camPams(rhs.camPams) {
    // intentionally empty
}


/*
 * vislib::graphics::RelativeCursor3D::~RelativeCursor3D
 */
vislib::graphics::RelativeCursor3D::~RelativeCursor3D(void) {
    // Do not delete this->cam
}


/*
 * vislib::graphics::RelativeCursor3D::Motion
 */
void vislib::graphics::RelativeCursor3D::Motion(SceneSpaceType tx,
        SceneSpaceType ty, SceneSpaceType tz, float rx, float ry, float rz) {

    this->translate.Set(tx, ty, tz);
    this->rotate.Set(rx, ry, rz);

    if (!(translate.IsNull() && rotate.IsNull())) { // trigger move events
        AbstractCursor::TriggerMoved();
    }
}


/*
 * vislib::graphics::RelativeCursor3D::RegisterCursorEvent
 */
void vislib::graphics::RelativeCursor3D::RegisterCursorEvent(
        AbstractCursor3DEvent *cursorEvent) {
    AbstractCursor::RegisterCursorEvent(cursorEvent);
}


/*
 * vislib::graphics::RelativeCursor3D::operator=
 */
vislib::graphics::RelativeCursor3D&
vislib::graphics::RelativeCursor3D::operator=(const RelativeCursor3D& rhs) {
    AbstractCursor::operator=(rhs);
    this->translate = rhs.translate;
    this->rotate = rhs.rotate;
    this->camPams = rhs.camPams;
    return *this;
}


/*
 * vislib::graphics::RelativeCursor3D::SetCamera
 */
void vislib::graphics::RelativeCursor3D::SetCameraParams(
        vislib::SmartPtr<CameraParameters> cameraParams) {
    this->camPams = cameraParams;
}
