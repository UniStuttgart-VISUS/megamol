/*
* CallOSPRayMaterial.cpp
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRay_plugin/CallOSPRayMaterial.h"
#include "vislib/IllegalParamException.h"

using namespace megamol::ospray;

// ################################
// ###### CallOSPRayMaterial ######
// ################################
/*
* megamol::ospray::CallOSPRayLight::CallOSPRayLight
*/
CallOSPRayMaterial::CallOSPRayMaterial(void) : isDirty(false) {
    // intentionally empty
}

/*
* megamol::ospray::CallOSPRayLight::~CallOSPRayLight
*/
CallOSPRayMaterial::~CallOSPRayMaterial(void) {
    //
}

/*
* megamol::ospray::CallOSPRayLight::operator=
*/
CallOSPRayMaterial& CallOSPRayMaterial::operator=(const CallOSPRayMaterial& rhs) {
    return *this;
}

void CallOSPRayMaterial::setMaterialContainer(std::shared_ptr<OSPRayMaterialContainer> mc) {
    this->materialContainer = mc;
}

std::shared_ptr<OSPRayMaterialContainer> CallOSPRayMaterial::getMaterialParameter() {
    if (!(*this)(0)) {
        throw vislib::IllegalParamException("Error in fillMaterialContainer", __FILE__, __LINE__);
    }
    return std::move(this->materialContainer);
}

bool CallOSPRayMaterial::InterfaceIsDirty() {
    if (this->isDirty) {
        this->isDirty = false;
        return true;
    }
    return false;
}

void CallOSPRayMaterial::setDirty() {
    this->isDirty = true;
}