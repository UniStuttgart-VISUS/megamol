/*
 * Cursor2D.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/graphics/Cursor2D.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/graphics/AbstractCursor2DEvent.h"

#include <stdio.h>


/*
 * vislib::graphics::Cursor2D::Cursor2D
 */
vislib::graphics::Cursor2D::Cursor2D(void) : AbstractCursor(), 
        x(static_cast<ImageSpaceType>(0)), y(static_cast<ImageSpaceType>(0)), 
        prevX(static_cast<ImageSpaceType>(0)), 
        prevY(static_cast<ImageSpaceType>(0)) { 
}


/*
 * vislib::graphics::Cursor2D::Cursor2D
 */
vislib::graphics::Cursor2D::Cursor2D(const Cursor2D& rhs) 
        : AbstractCursor(rhs), x(rhs.x), y(rhs.y), prevX(rhs.prevX), 
        prevY(rhs.prevY) {
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
void vislib::graphics::Cursor2D::SetPosition(ImageSpaceType x, ImageSpaceType y, bool flipY, float height) {
    bool moved = !math::IsEqual(this->x, x) || !math::IsEqual(this->y, y);

    if (moved) {
        this->prevX = this->x;
        this->prevY = this->y;
    }
    this->x = x;
    this->y = y;

    if (flipY) {
        this->y = height - static_cast<ImageSpaceType>(1) - this->y;
    }

    if (moved) { // trigger move events
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
    this->prevX = rhs.prevX;
    this->prevY = rhs.prevY;
    return *this;
}
