/*
 * AbstractParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/AbstractParamSlot.h"

using namespace megamol::core::param;


/*
 * AbstractParam::~AbstractParam
 */
AbstractParam::~AbstractParam(void) {
    this->slot = NULL; // DO NOT DELETE
}


/*
 * AbstractParam::AbstractParam
 */
AbstractParam::AbstractParam(void) : slot(NULL), hash(0), has_changed(false) {
    // intentionally empty
}


/*
 * AbstractParam::isSlotPublic
 */
bool AbstractParam::isSlotPublic(void) const {
    return (this->slot == NULL) ? false : (this->slot->isSlotAvailable());
}


/*
 * AbstractParam::setDirty
 */
void AbstractParam::setDirty(void) {
    if (this->slot == NULL)
        return; // fail silently
    this->slot->update();
}
