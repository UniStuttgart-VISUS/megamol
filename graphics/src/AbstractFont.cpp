/*
 * AbstractFont.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractFont.h"
#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"


/*
 * vislib::graphics::AbstractFont::~AbstractFont
 */
vislib::graphics::AbstractFont::~AbstractFont(void) {
    // Deinitialise must be called from the dtor of the implementing class.
    ASSERT(this->initialised == false);
}


/*
 * vislib::graphics::AbstractFont::Deinitialise
 */
void vislib::graphics::AbstractFont::Deinitialise(void) {
    if (this->initialised) {
        this->deinitialise();
        this->initialised = false;
    }
}


/*
 * vislib::graphics::AbstractFont::Initialise
 */
bool vislib::graphics::AbstractFont::Initialise(void) {
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
vislib::graphics::AbstractFont::AbstractFont(void) : initialised(false),
        size(1.0f), flipY(false) {
    // intentionally empty
}
