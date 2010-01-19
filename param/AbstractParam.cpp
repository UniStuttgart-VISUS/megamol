/*
 * AbstractParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AbstractParam.h"
#include "AbstractParamSlot.h"

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
AbstractParam::AbstractParam(bool visible) : slot(NULL),
        visible(visible) {
    // intentionally empty
}


/*
 * AbstractParam::isSlotPublic
 */
bool AbstractParam::isSlotPublic(void) const {
    return (this->slot == NULL) ? false
        : (this->slot->isSlotAvailable());
}


/*
 * AbstractParam::setDirty
 */
void AbstractParam::setDirty(void) {
    if (this->slot == NULL) return; // fail silently
    this->slot->update();
}
