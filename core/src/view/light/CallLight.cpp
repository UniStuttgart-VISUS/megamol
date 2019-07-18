/*
 * CallLight.cpp
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/light/CallLight.h"
#include "vislib/IllegalParamException.h"

using namespace megamol::core::view::light;

/*
 * LightContainer::LightContainer
 */
LightContainer::LightContainer()
    : // General light parameters
    lightType(lightenum::NONE)
    , lightColor(NULL)
    , lightIntensity(0.0f)
    ,
    // Distant light parameters
    dl_direction(NULL)
    , dl_angularDiameter(0.0f)
    , dl_eye_direction(false)
    ,
    // point light paramenters
    pl_position(NULL)
    , pl_radius(0.0f)
    ,
    // spot light parameters
    sl_position(NULL)
    , sl_direction(NULL)
    , sl_openingAngle(0.0f)
    , sl_penumbraAngle(0.0f)
    , sl_radius(0.0f)
    ,
    // quad light parameters
    ql_position(NULL)
    , ql_edgeOne(NULL)
    , ql_edgeTwo(NULL)
    ,
    // hdri light parameters
    hdri_up(NULL)
    , hdri_direction(NULL)
    , hdri_evnfile("")
    ,
    // tracks the existence of the light module
    isValid(false)
    , dataChanged(false) {}

LightContainer::~LightContainer() { this->isValid = false; }


// #############################
// ###### CallLight ######
// #############################
/*
 * megamol::core::view::light::CallLight::CallLight
 */
CallLight::CallLight() {
    // intentionally empty
}

/*
 * megamol::core::view::light::CallLight::setLightMap
 */
void CallLight::setLightMap(LightMap* lm) { this->lightMap = lm; }

/*
 * megamol::core::view::light::CallLight::~CallLight
 */
CallLight::~CallLight(void) {
    //
}

/*
 * megamol::core::view::light::CallLight::operator=
 */
CallLight& CallLight::operator=(const CallLight& rhs) {
    lightMap = rhs.lightMap;
    return *this;
}


/*
 * megamol::core::view::light::CallLight::addLight
 */
void CallLight::addLight(LightContainer& lc) {
    if (lc.isValid) {
        if (this->lightMap != NULL) {
            // this->lightMap->insert_or_assign(this, lc); // C++17
            this->lightMap->operator[](this) = lc;
        } else {
            throw vislib::IllegalParamException("Error: no lightMap set.", __FILE__, __LINE__);
        }
    }
}

/*
 * megamol::core::view::light::CallLight::fillLightMap
 */
void CallLight::fillLightMap() {
    if (!(*this)(0)) {
        throw vislib::IllegalParamException("Error in fillLightMap", __FILE__, __LINE__);
    }
}
