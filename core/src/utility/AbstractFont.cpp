/*
 * AbstractFont.cpp
 *
 * Copyright (C) 2006 - 2018 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 *
 * This implementation is based on "vislib/graphics/AbstractFont.h"
 */

#include "mmcore/utility/AbstractFont.h"


 /*
 * megamol::core::utility::AbstractFont::AbstractFont
 */
megamol::core::utility::AbstractFont::AbstractFont(void) : initialised(false), size(1.0f), flipY(false) {
    // nothing to do here ...
}


/*
 * megamol::core::utility::AbstractFont::~AbstractFont
 */
megamol::core::utility::AbstractFont::~AbstractFont(void) {
    // Deinitialise must be called from the dtor of the implementing class.
    ASSERT(this->initialised == false);
}


/*
 * megamol::core::utility::AbstractFont::Initialise
 */
bool megamol::core::utility::AbstractFont::Initialise(megamol::core::CoreInstance *core) {
    if (!this->initialised) {
        if (!this->initialise(core)) {
            return false;
        }
        this->initialised = true;
    }
    return true;
}


/*
* megamol::core::utility::AbstractFont::Deinitialise
*/
void megamol::core::utility::AbstractFont::Deinitialise(void) {
    if (this->initialised) {
        this->deinitialise();
        this->initialised = false;
    }
}


/*
 * megamol::core::utility::AbstractFont::LineHeight
 */
float megamol::core::utility::AbstractFont::LineHeight(float size) const {
    return size;
}


/*
 * megamol::core::utility::AbstractFont::SetFlipY
 */
void megamol::core::utility::AbstractFont::SetFlipY(bool flipY) {
    this->flipY = flipY;
}


/*
 * megamol::core::utility::AbstractFont::SetSize
 */
void megamol::core::utility::AbstractFont::SetSize(float size) {
    if (size < 0.0f) {
        throw vislib::IllegalParamException("size", __FILE__, __LINE__);
    }
    this->size = size;
}


