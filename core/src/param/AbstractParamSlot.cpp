/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/param/AbstractParamSlot.h"

#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"

using namespace megamol::core::param;


/*
 * AbstractParamSlot::AbstractParamSlot
 */
AbstractParamSlot::AbstractParamSlot() : dirty(false), param() {
    // intentionally empty
}


/*
 * AbstractParamSlot::~AbstractParamSlot
 */
AbstractParamSlot::~AbstractParamSlot() {
    this->param = nullptr; // SmartPtr will clean up
}


/*
 * AbstractParamSlot::SetParameter
 */
void AbstractParamSlot::SetParameter(AbstractParam* param) {
    if (param == nullptr) {
        throw vislib::IllegalParamException("param", __FILE__, __LINE__);
    }
    if (this->isSlotAvailable()) {
        throw vislib::IllegalStateException(
            "Slot must not be public when setting a parameter object.", __FILE__, __LINE__);
    }
    if (this->param != nullptr) {
        throw vislib::IllegalStateException(
            "There already is an parameter object set for this slot.", __FILE__, __LINE__);
    }
    if (param->slot != nullptr) {
        throw vislib::IllegalParamException("Parameter object already assigned to a slot.", __FILE__, __LINE__);
    }

    this->param = std::shared_ptr<AbstractParam>(param);
    this->param->slot = this;
}


/*
 * AbstractParamSlot::update
 */
void AbstractParamSlot::update() {
    this->dirty = true;
}
