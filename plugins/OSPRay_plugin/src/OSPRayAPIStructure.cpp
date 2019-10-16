/*
* OSPRayAPIStructure.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayAPIStructure.h"
#include "OSPRay_plugin/CallOSPRayAPIObject.h"
#include "vislib/sys/Log.h"
#include "mmcore/Call.h"


using namespace megamol::ospray;


OSPRayAPIStructure::OSPRayAPIStructure(void) :
    AbstractOSPRayStructure(),
    getDataSlot("getdata", "Connects to the data source")
{
    this->getDataSlot.SetCompatibleCall<CallOSPRayAPIObjectDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);
}


bool OSPRayAPIStructure::readData(megamol::core::Call &call) {

    // fill material container
    this->processMaterial();

    // read Data, calculate  shape parameters, fill data vectors
    CallOSPRayStructure *os = dynamic_cast<CallOSPRayStructure*>(&call);
    CallOSPRayAPIObject *cd = this->getDataSlot.CallAs<CallOSPRayAPIObject>();

    
    this->structureContainer.dataChanged = false;
    if (cd == NULL) return false;
    if (!(*cd)(2)) return false; // get dirty
    cd->SetTimeStamp(os->getTime());
    cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    if (this->datahash != cd->DataHash() || this->frameID != static_cast<size_t>(os->getTime()) || this->InterfaceIsDirty() || cd->isDirty()) {
        this->datahash = cd->DataHash();
        this->time = os->getTime();
        this->frameID = static_cast<size_t>(os->getTime());
        this->structureContainer.dataChanged = true;
        cd->resetDirty();
    } else {
        return true;
    }

    if (!(*cd)(1)) return false;
    if (!(*cd)(0)) return false;



    // Write stuff into the structureContainer
    this->structureContainer.type = structureTypeEnum::OSPRAY_API_STRUCTURES;

    switch (cd->getStructureType()) {
    case structureTypeEnum::GEOMETRY:
        this->structureContainer.ospStructures = std::make_pair<std::vector<void*>, structureTypeEnum>(cd->getAPIObjects(), structureTypeEnum::GEOMETRY);
        break;
    case structureTypeEnum::VOLUME:
        this->structureContainer.ospStructures = std::make_pair<std::vector<void*>, structureTypeEnum>(cd->getAPIObjects(), structureTypeEnum::VOLUME);
        break;
    case structureTypeEnum::UNINITIALIZED:
        vislib::sys::Log::DefaultLog.WriteError("OSPRay API structure type is not set.");
        return false;
    }

    return true;
}


OSPRayAPIStructure::~OSPRayAPIStructure() {
    this->Release();
}

bool OSPRayAPIStructure::create() {
    return true;
}

void OSPRayAPIStructure::release() {

}

/*
ospray::OSPRaySphereGeometry::InterfaceIsDirty()
*/
bool OSPRayAPIStructure::InterfaceIsDirty() {
    return false;
}



bool OSPRayAPIStructure::getExtends(megamol::core::Call &call) {
    CallOSPRayStructure *os = dynamic_cast<CallOSPRayStructure*>(&call);
    CallOSPRayAPIObject *cd = this->getDataSlot.CallAs<CallOSPRayAPIObject>();
    
    if (cd == NULL) return false;
    if (os->getTime() > cd->FrameCount()) {
        cd->SetFrameID(cd->FrameCount() - 1, true);  // isTimeForced flag set to true
    } else {
        cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    }

    if (!(*cd)(1)) return false;

    this->extendContainer.boundingBox = std::make_shared<core::BoundingBoxes_2>(cd->AccessBoundingBoxes());
    this->extendContainer.timeFramesCount = cd->FrameCount();
    this->extendContainer.isValid = true;

    return true;
}