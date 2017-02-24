/*
* CallOSPRayStructure.cpp
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "CallOSPRayStructure.h"
#include "vislib/sys/Log.h"

using namespace megamol::ospray;


OSPRayStructureContainer::OSPRayStructureContainer() :
    isValid(false) { }

OSPRayStructureContainer::~OSPRayStructureContainer() {
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
            vislib::sys::Log::DefaultLog.WriteError("Error: no stuctureMap set.");
        }
    }
}

void CallOSPRayStructure::fillStructureMap() {
    if (!(*this)(0)) {
        vislib::sys::Log::DefaultLog.WriteError("Error in fillStructureMap");
    }
}

void CallOSPRayStructure::setTime(float time) {
    this->time = time;
}

float CallOSPRayStructure::getTime() {
    return this->time;
}

