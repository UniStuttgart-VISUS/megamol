/*
* CallOSPRayAPIObject.cpp
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRay_plugin/CallOSPRayAPIObject.h"
#include "vislib/IllegalParamException.h"

using namespace megamol::ospray;

/*
* megamol::ospray::CallOSPRayAPIObject::CallOSPRayAPIObject
*/
CallOSPRayAPIObject::CallOSPRayAPIObject() {
// intentionally empty
}


/*
* megamol::ospray::CallOSPRayAPIObject::~CallOSPRayAPIObject
*/
CallOSPRayAPIObject::~CallOSPRayAPIObject(void) {
    //
}


void CallOSPRayAPIObject::setAPIObject(void* api_obj) {
    this->api_obj = api_obj;
}

void* CallOSPRayAPIObject::getAPIObject() {
    return this->api_obj;
}

void megamol::ospray::CallOSPRayAPIObject::setStructureType(structureTypeEnum strtype) {
    this->type = strtype;
}

structureTypeEnum megamol::ospray::CallOSPRayAPIObject::getStructureType() {
    return this->type;
}
