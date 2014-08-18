/*
 * Cursor2DRectLasso.cpp
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/Cursor2DRectLasso.h"
#include "vislib/assert.h"
#include "vislib/Cursor2D.h"


/*
 * vislib::graphics::Cursor2DRectLasso::Cursor2DRectLasso
 */
vislib::graphics::Cursor2DRectLasso::Cursor2DRectLasso(void)
        : AbstractCursor2DEvent(), drag(false), rect() {
    // Intentionally empty
}


/*
 * vislib::graphics::Cursor2DRectLasso::~Cursor2DRectLasso
 */
vislib::graphics::Cursor2DRectLasso::~Cursor2DRectLasso(void) {
    // Intentionally empty
}


/*
 * vislib::graphics::Cursor2DRectLasso::Clear
 */
void vislib::graphics::Cursor2DRectLasso::Clear(void) {
    this->rect.SetWidth(0);
    this->rect.SetHeight(0);
}


/*
 * vislib::graphics::Cursor2DRectLasso::Trigger
 */
void vislib::graphics::Cursor2DRectLasso::Trigger(
        vislib::graphics::AbstractCursor *caller,
        vislib::graphics::AbstractCursorEvent::TriggerReason reason,
        unsigned int param) {
    vislib::graphics::Cursor2D *cursor 
        = dynamic_cast<vislib::graphics::Cursor2D*>(caller);
    ASSERT(cursor != NULL);

    if (reason == REASON_BUTTON_DOWN) {
        this->drag = true;
        this->rect.Set(cursor->X(), cursor->Y(), cursor->X(), cursor->Y());

    } else if (reason == REASON_BUTTON_UP) {
        this->drag = false;

    } else if ((reason == REASON_MOVE) && this->drag) {
        this->rect.SetRight(cursor->X());
        this->rect.SetTop(cursor->Y());

    }

}
