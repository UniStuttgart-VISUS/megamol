/**
 * MegaMol
 * Copyright (c) 2008, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/param/AbstractParam.h"

#include "mmcore/param/AbstractParamSlot.h"

using namespace megamol::core::param;


/*
 * AbstractParam::~AbstractParam
 */
AbstractParam::~AbstractParam() {
    this->slot = nullptr; // DO NOT DELETE
}


/*
 * AbstractParam::AbstractParam
 */
AbstractParam::AbstractParam() : slot(nullptr), hash(0) {
    // intentionally empty
}


/*
 * AbstractParam::isSlotPublic
 */
bool AbstractParam::isSlotPublic() const {
    return (this->slot == nullptr) ? false : (this->slot->isSlotAvailable());
}


/*
 * AbstractParam::setDirty
 */
void AbstractParam::setDirty() {
    if (this->slot == nullptr)
        return; // fail silently
    this->slot->update();
}
