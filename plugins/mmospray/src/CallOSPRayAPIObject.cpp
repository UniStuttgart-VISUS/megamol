/*
 * CallOSPRayAPIObject.cpp
 *
 * Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "mmospray/CallOSPRayAPIObject.h"
#include "stdafx.h"
#include "vislib/IllegalParamException.h"

using namespace megamol::ospray;

/*
 * megamol::ospray::CallOSPRayAPIObject::CallOSPRayAPIObject
 */
CallOSPRayAPIObject::CallOSPRayAPIObject()
        : datahash(0)
        , unlocker(NULL)
        , forceFrame(false)
        , frameCnt(0)
        , frameID(0)
        , bboxs() {
    // intentionally empty
}


/*
 * megamol::ospray::CallOSPRayAPIObject::~CallOSPRayAPIObject
 */
CallOSPRayAPIObject::~CallOSPRayAPIObject(void) {
    this->Unlock();
}


void CallOSPRayAPIObject::setAPIObjects(std::vector<void*> api_obj) {
    this->api_obj = api_obj;
}

std::vector<void*> CallOSPRayAPIObject::getAPIObjects() {
    return this->api_obj;
}

void megamol::ospray::CallOSPRayAPIObject::setStructureType(structureTypeEnum strtype) {
    this->type = strtype;
}

structureTypeEnum megamol::ospray::CallOSPRayAPIObject::getStructureType() {
    return this->type;
}

void megamol::ospray::CallOSPRayAPIObject::resetDirty() {
    this->dirtyFlag = false;
}

void megamol::ospray::CallOSPRayAPIObject::setDirty() {
    this->dirtyFlag = true;
}

bool megamol::ospray::CallOSPRayAPIObject::isDirty() {
    return this->dirtyFlag;
}

/*
 * CallOSPRayAPIObject::operator=
 */
megamol::ospray::CallOSPRayAPIObject& megamol::ospray::CallOSPRayAPIObject::operator=(
    const megamol::ospray::CallOSPRayAPIObject& rhs) {
    megamol::ospray::CallOSPRayAPIObject::operator=(rhs);
    this->forceFrame = rhs.forceFrame;
    this->frameCnt = rhs.frameCnt;
    this->frameID = rhs.frameID;
    this->bboxs = rhs.bboxs;
    this->datahash = rhs.datahash;
    this->unlocker = rhs.unlocker;
    return *this;
}
