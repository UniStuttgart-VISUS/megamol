/*
 * AbstractFont.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */
 /*
 * This is a copy of "vislib/graphics/AbstractFont.h"
 */

#include "mmcore/view/special/AbstractFont.h"

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"


/*
 * megamol::core::view::special::AbstractFont::~AbstractFont
 */
megamol::core::view::special::AbstractFont::~AbstractFont(void) {
    // Deinitialise must be called from the dtor of the implementing class.
    ASSERT(this->initialised == false);
}


/*
 * megamol::core::view::special::AbstractFont::Deinitialise
 */
void megamol::core::view::special::AbstractFont::Deinitialise(void) {
    if (this->initialised) {
        this->deinitialise();
        this->initialised = false;
    }
}


/*
 * megamol::core::view::special::AbstractFont::Initialise
 */
bool megamol::core::view::special::AbstractFont::Initialise(void) {
    if (!this->initialised) {
        if (!this->initialise()) {
            return false;
        }
        this->initialised = true;
    }
    return true;
}


/*
 * megamol::core::view::special::AbstractFont::LineHeight
 */
float megamol::core::view::special::AbstractFont::LineHeight(float size) const {
    return size;
}


/*
 * megamol::core::view::special::AbstractFont::SetFlipY
 */
void megamol::core::view::special::AbstractFont::SetFlipY(bool flipY) {
    this->flipY = flipY;
}


/*
 * megamol::core::view::special::AbstractFont::SetSize
 */
void megamol::core::view::special::AbstractFont::SetSize(float size) {
    if (size < 0.0f) {
        throw vislib::IllegalParamException("size", __FILE__, __LINE__);
    }
    this->size = size;
}


/*
 * megamol::core::view::special::AbstractFont::AbstractFont
 */
megamol::core::view::special::AbstractFont::AbstractFont(void) : initialised(false),
        size(1.0f), flipY(false) {
    // intentionally empty
}
