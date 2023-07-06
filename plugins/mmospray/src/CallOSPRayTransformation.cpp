/*
 * CallOSPRayTransformation.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "mmospray/CallOSPRayTransformation.h"
#include "vislib/IllegalParamException.h"

namespace megamol::ospray {

// ################################
// ###### CallOSPRayTransformation ######
// ################################

CallOSPRayTransformation::CallOSPRayTransformation() : _isDirty(false) {
    // intentionally empty
}

CallOSPRayTransformation::~CallOSPRayTransformation() {
    //
}

CallOSPRayTransformation& CallOSPRayTransformation::operator=(const CallOSPRayTransformation& rhs) {
    return *this;
}

void CallOSPRayTransformation::setTransformationContainer(std::shared_ptr<OSPRayTransformationContainer> tc) {

    this->_transformationContainer = tc;
}

std::shared_ptr<OSPRayTransformationContainer> CallOSPRayTransformation::getTransformationParameter() {

    if (!(*this)(0)) {
        throw vislib::IllegalParamException("[CallOSPRayTransformation]: Illegal parameter", __FILE__, __LINE__);
    }
    return std::move(this->_transformationContainer);
}

bool CallOSPRayTransformation::InterfaceIsDirty() {
    if (this->_isDirty) {
        this->_isDirty = false;
        return true;
    }
    return false;
}

void CallOSPRayTransformation::setDirty() {
    this->_isDirty = true;
}

} // namespace megamol::ospray
