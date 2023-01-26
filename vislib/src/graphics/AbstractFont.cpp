/*
 * AbstractFont.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "vislib/graphics/AbstractFont.h"
#include "vislib/IllegalParamException.h"
#include "vislib/assert.h"


/*
 * vislib::graphics::AbstractFont::~AbstractFont
 */
vislib::graphics::AbstractFont::~AbstractFont() {
    // Deinitialise must be called from the dtor of the implementing class.
    ASSERT(this->initialised == false);
}


/*
 * vislib::graphics::AbstractFont::Deinitialise
 */
void vislib::graphics::AbstractFont::Deinitialise() {
    if (this->initialised) {
        this->deinitialise();
        this->initialised = false;
    }
}


/*
 * vislib::graphics::AbstractFont::Initialise
 */
bool vislib::graphics::AbstractFont::Initialise() {
    if (!this->initialised) {
        if (!this->initialise()) {
            return false;
        }
        this->initialised = true;
    }
    return true;
}


/*
 * vislib::graphics::AbstractFont::LineHeight
 */
float vislib::graphics::AbstractFont::LineHeight(float size) const {
    return size;
}


/*
 * vislib::graphics::AbstractFont::SetFlipY
 */
void vislib::graphics::AbstractFont::SetFlipY(bool flipY) {
    this->flipY = flipY;
}


/*
 * vislib::graphics::AbstractFont::SetSize
 */
void vislib::graphics::AbstractFont::SetSize(float size) {
    if (size < 0.0f) {
        throw vislib::IllegalParamException("size", __FILE__, __LINE__);
    }
    this->size = size;
}


/*
 * vislib::graphics::AbstractFont::AbstractFont
 */
vislib::graphics::AbstractFont::AbstractFont() : initialised(false), size(1.0f), flipY(false) {
    // intentionally empty
}
