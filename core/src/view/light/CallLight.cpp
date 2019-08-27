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
