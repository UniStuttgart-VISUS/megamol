/*
* CallOSPRayStructure.cpp
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "CallOSPRayStructure.h"
#include "vislib/IllegalParamException.h"

using namespace megamol::ospray;


OSPRayStructureContainer::OSPRayStructureContainer() :
    isValid(false) { }

OSPRayStructureContainer::~OSPRayStructureContainer() {
    //
}

OSPRayExtendContainer::OSPRayExtendContainer() :
    isValid(false) {
}

OSPRayExtendContainer::~OSPRayExtendContainer() {
    //
}


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

void CallOSPRayStructure::setStructureMap(OSPRayStrcutrureMap *sm) {
    this->structureMap = sm;
}


void CallOSPRayStructure::addStructure(OSPRayStructureContainer &sc) {
    if (sc.isValid) {
        if (this->structureMap != NULL) {
            this->structureMap->insert_or_assign(this, sc);
        } else {
            throw vislib::IllegalParamException("Error: no stuctureMap set.", __FILE__, __LINE__);
        }
    }
}

void CallOSPRayStructure::fillStructureMap() {
    if (!(*this)(0)) {
        throw vislib::IllegalParamException("Error in fillStructureMap", __FILE__, __LINE__);
    }
}


void CallOSPRayStructure::setExtendMap(OSPRayExtendMap *em) {
    this->extendMap = em;
}

void CallOSPRayStructure::addExtend(OSPRayExtendContainer &ec) {
    if (this->extendMap != NULL) {
        this->extendMap->insert_or_assign(this, ec);
    } else {
        throw vislib::IllegalParamException("Error: no bounding box map set.", __FILE__, __LINE__);
    }
}

void CallOSPRayStructure::fillExtendMap() {
    if (!(*this)(1)) {
        throw vislib::IllegalParamException("Error in fillExtendMap", __FILE__, __LINE__);
    }
}


void CallOSPRayStructure::setTime(float time) {
    this->time = time;
}

float CallOSPRayStructure::getTime() {
    return this->time;
}

