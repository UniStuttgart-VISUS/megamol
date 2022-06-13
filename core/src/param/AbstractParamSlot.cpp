/*
 * AbstractParam.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/param/AbstractParamSlot.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"

using namespace megamol::core::param;


/*
 * AbstractParamSlot::AbstractParamSlot
 */
AbstractParamSlot::AbstractParamSlot(void) : dirty(false), param() {
    // intentionally empty
}


/*
 * AbstractParamSlot::~AbstractParamSlot
 */
AbstractParamSlot::~AbstractParamSlot(void) {
    this->param = NULL; // SmartPtr will clean up
}


/*
 * AbstractParamSlot::SetParameter
 */
void AbstractParamSlot::SetParameter(AbstractParam* param) {
    if (param == NULL) {
        throw vislib::IllegalParamException("param", __FILE__, __LINE__);
    }
    if (this->isSlotAvailable()) {
        throw vislib::IllegalStateException(
            "Slot must not be public when setting a parameter object.", __FILE__, __LINE__);
    }
    if (!this->param.IsNull()) {
        throw vislib::IllegalStateException(
            "There already is an parameter object set for this slot.", __FILE__, __LINE__);
    }
    if (param->slot != NULL) {
        throw vislib::IllegalParamException("Parameter object already assigned to a slot.", __FILE__, __LINE__);
    }

    this->param = param;
    this->param->slot = this;
}


/*
 * AbstractParamSlot::update
 */
void AbstractParamSlot::update(void) {
    this->dirty = true;
}
