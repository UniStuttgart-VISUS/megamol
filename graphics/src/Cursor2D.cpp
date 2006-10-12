/*
 * Cursor2D.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/Cursor2D.h"
#include "vislib/Camera.h"
#include "vislib/mathfunctions.h"
#include "vislib/AbstractCursor2DEvent.h"

#include <stdio.h>


/*
 * vislib::graphics::Cursor2D::Cursor2D
 */
vislib::graphics::Cursor2D::Cursor2D(void) : AbstractCursor(), cam(NULL) {
}


/*
 * vislib::graphics::Cursor2D::Cursor2D
 */
vislib::graphics::Cursor2D::Cursor2D(const Cursor2D& rhs) 
    : AbstractCursor(rhs), x(rhs.x), y(rhs.y), cam(rhs.cam) {
}


/*
 * vislib::graphics::Cursor2D::~Cursor2D
 */
vislib::graphics::Cursor2D::~Cursor2D(void) {
    // Do not delete this->cam
}


/*
 * vislib::graphics::Cursor2D::SetPosition
 */
void vislib::graphics::Cursor2D::SetPosition(ImageSpaceType x, ImageSpaceType y, bool flipY) {
    bool unmoved = math::IsEqual(this->x, x) && math::IsEqual(this->y, y);

    this->x = x;
    this->y = y;

    if (flipY && this->cam) {
        this->y = this->cam->GetVirtualHeight() - static_cast<ImageSpaceType>(1) - this->y;
    }

    if (!unmoved) { // trigger move events
        AbstractCursor::TriggerMoved();
    }
}


/*
 * vislib::graphics::Cursor2D::RegisterCursorEvent
 */
void vislib::graphics::Cursor2D::RegisterCursorEvent(AbstractCursor2DEvent *cursorEvent) {
    AbstractCursor::RegisterCursorEvent(cursorEvent);
}


/*
 * vislib::graphics::Cursor2D::operator=
 */
vislib::graphics::Cursor2D& vislib::graphics::Cursor2D::operator=(const Cursor2D& rhs) {
    AbstractCursor::operator=(rhs);
    this->x = rhs.x;
    this->y = rhs.y; 
    this->cam = rhs.cam;
    return *this;
}


/*
 * vislib::graphics::Cursor2D::SetCamera
 */
void vislib::graphics::Cursor2D::SetCamera(Camera *camera) {
    this->cam = camera;
}
