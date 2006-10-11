/*
 * Cursor2D.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "vislib/Cursor2D.h"
#include "vislib/mathfunctions.h"
#include "vislib/AbstractCursor2DEvent.h"

#include <stdio.h>


/*
 * vislib::graphics::Cursor2D::Cursor2D
 */
vislib::graphics::Cursor2D::Cursor2D(void) : AbstractCursor() {
}


/*
 * vislib::graphics::Cursor2D::Cursor2D
 */
vislib::graphics::Cursor2D::Cursor2D(const Cursor2D& rhs) : AbstractCursor(rhs) {
    this->width = rhs.width;
    this->height = rhs.height;
    this->x = rhs.x;
    this->y = rhs.y;
}


/*
 * vislib::graphics::Cursor2D::~Cursor2D
 */
vislib::graphics::Cursor2D::~Cursor2D(void) {
}


/*
 * vislib::graphics::Cursor2D::SetSize
 */
void vislib::graphics::Cursor2D::SetSize(CursorSpaceType width, CursorSpaceType height) {
    this->width = width;
    this->height = height;
}


/*
 * vislib::graphics::Cursor2D::SetPosition
 */
void vislib::graphics::Cursor2D::SetPosition(CursorSpaceType x, CursorSpaceType y) {
    bool unmoved = math::IsEqual(this->x, x) && math::IsEqual(this->y, y);

    this->x = x;
    this->y = y;

    printf("Cursor2D::SetPosition(%f, %f)\n", this->x, this->y);

    if (unmoved) {
        // trigger move events
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
    this->width = rhs.width;
    this->height = rhs.height;
    this->x = rhs.x;
    this->y = rhs.y; 
    return *this;
}
