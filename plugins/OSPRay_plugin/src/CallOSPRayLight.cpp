/*
* CallOSPRayLight.cpp
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "CallOSPRayLight.h"
#include "vislib/IllegalParamException.h"

using namespace megamol::ospray;

OSPRayLightContainer::OSPRayLightContainer() :
    // General light parameters
    lightType(lightenum::NONE),
    lightColor(NULL),
    lightIntensity(0.0f),
    // Distant light parameters
    dl_direction(NULL),
    dl_angularDiameter(0.0f),
    dl_eye_direction(false),
    // point light paramenters
    pl_position(NULL),
    pl_radius(0.0f),
    // spot light parameters
    sl_position(NULL),
    sl_direction(NULL),
    sl_openingAngle(0.0f),
    sl_penumbraAngle(0.0f),
    sl_radius(0.0f),
    // quad light parameters
    ql_position(NULL),
    ql_edgeOne(NULL),
    ql_edgeTwo(NULL),
    // hdri light parameters
    hdri_up(NULL),
    hdri_direction(NULL),
    hdri_evnfile(""),
    // tracks the existence of the light module
    isValid(false),
    dataChanged(false) { }

OSPRayLightContainer::~OSPRayLightContainer() {
    this->isValid = false;
}



// #############################
// ###### CallOSPRayLight ######
// #############################
/*
* megamol::ospray::CallOSPRayLight::CallOSPRayLight
*/
CallOSPRayLight::CallOSPRayLight() { 
// intentionally empty
}

void CallOSPRayLight::setLightMap(OSPRayLightMap *lm) {
    this->lightMap = lm;
}

/*
* megamol::ospray::CallOSPRayLight::~CallOSPRayLight
*/
CallOSPRayLight::~CallOSPRayLight(void) {
    //
}

/*
* megamol::ospray::CallOSPRayLight::operator=
*/
CallOSPRayLight& CallOSPRayLight::operator=(const CallOSPRayLight& rhs) {
    lightMap = rhs.lightMap;
    return *this;
}

void CallOSPRayLight::addLight(OSPRayLightContainer &lc) {
    if (lc.isValid) {
        if (this->lightMap != NULL) {
            //this->lightMap->insert_or_assign(this, lc); // C++17
            this->lightMap->operator[](this) = lc;
        } else {
            throw vislib::IllegalParamException("Error: no lightMap set.", __FILE__, __LINE__);
        }
    }
}

void CallOSPRayLight::fillLightMap() {
    if (!(*this)(0)) {
        throw vislib::IllegalParamException("Error in fillLightMap", __FILE__, __LINE__);
    }
}

