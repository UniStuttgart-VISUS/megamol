/*
 * CallOSPRayStructure.cpp
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "mmospray/CallOSPRayStructure.h"
#include "vislib/IllegalParamException.h"

using namespace megamol::ospray;


// #################################
// ###### CallOSPRayStructure ######
// #################################

/*
 * megamol::ospray::CallOSPRayStructure::CallOSPRayStructure
 */
CallOSPRayStructure::CallOSPRayStructure() {
    // intentionally empty
}

/*
 * megamol::ospray::CallOSPRayStructure::~CallOSPRayStructure
 */
CallOSPRayStructure::~CallOSPRayStructure(void) {
    this->structureMap = NULL;
}

/*
 * megamol::ospray::CallOSPRayLight::operator=
 */
CallOSPRayStructure& CallOSPRayStructure::operator=(const CallOSPRayStructure& rhs) {
    this->structureMap = rhs.structureMap;
    this->time = rhs.time;
    this->extendMap = rhs.extendMap;
    return *this;
}

void CallOSPRayStructure::setStructureMap(OSPRayStrcutrureMap* sm) {
    structureMap = sm;
}


void CallOSPRayStructure::addStructure(OSPRayStructureContainer* sc) {
    if (sc->isValid) {
        if (this->structureMap != NULL) {
            //this->structureMap->insert_or_assign(this, sc); // C++17
            this->structureMap->operator[](this) = sc;
        } else {
            throw vislib::IllegalParamException("Error: no stuctureMap set.", __FILE__, __LINE__);
        }
    }
}

bool CallOSPRayStructure::fillStructureMap() {
    return (*this)(0);
}


void CallOSPRayStructure::setExtendMap(OSPRayExtendMap* em) {
    this->extendMap = em;
}


void CallOSPRayStructure::addExtend(OSPRayExtendContainer* ec) {
    if (extendMap != nullptr) {
        //this->extendMap->insert_or_assign(this, ec); // C++17
        this->extendMap->operator[](this) = ec;
    } else {
        throw vislib::IllegalParamException("Error: no bounding box map set.", __FILE__, __LINE__);
    }
}


bool CallOSPRayStructure::fillExtendMap() {
    return (*this)(1);
}


void CallOSPRayStructure::setTime(float time) {
    this->time = time;
}


float CallOSPRayStructure::getTime() {
    return this->time;
}
