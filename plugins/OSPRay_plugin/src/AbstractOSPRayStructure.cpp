/*
* AbstractOSPRayStructure.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRay_plugin/AbstractOSPRayStructure.h"

using namespace megamol::ospray;


AbstractOSPRayStructure::AbstractOSPRayStructure(void) : 
    megamol::core::Module(),
    deployStructureSlot("deployStructureSlot", "Connects to the OSPRayRenderer or another OSPRayStructure"),
    getStructureSlot("getStructureSlot", "Connects to the another OSPRayStructure"),
    getMaterialSlot("getMaterialSlot", "Connects to an OSPRayMaterial") {

    this->deployStructureSlot.SetCallback(CallOSPRayStructure::ClassName(), CallOSPRayStructure::FunctionName(0), &AbstractOSPRayStructure::getStructureCallback);
    this->deployStructureSlot.SetCallback(CallOSPRayStructure::ClassName(), CallOSPRayStructure::FunctionName(1), &AbstractOSPRayStructure::getExtendsCallback);
    this->MakeSlotAvailable(&this->deployStructureSlot);

    this->getStructureSlot.SetCompatibleCall<CallOSPRayStructureDescription>();
    this->MakeSlotAvailable(&this->getStructureSlot);

    this->getMaterialSlot.SetCompatibleCall<CallOSPRayMaterialDescription>();
    this->MakeSlotAvailable(&this->getMaterialSlot);

    this->structureContainer.isValid = true;
    this->time = -1.0f;
}

AbstractOSPRayStructure::~AbstractOSPRayStructure(void) {
    this->structureContainer.isValid = false;
    this->Release();
}


/*
ospray::OSPRaySphereGeometry::getStructureCallback
*/
bool AbstractOSPRayStructure::getStructureCallback(megamol::core::Call& call) {
    CallOSPRayStructure *os_in = dynamic_cast<CallOSPRayStructure*>(&call);
    CallOSPRayStructure *os_out = this->getStructureSlot.CallAs<CallOSPRayStructure>();

    if (os_in != NULL) {
        if (!this->readData(call)) return false;
        os_in->addStructure(this->structureContainer);
    }

    if (os_out != NULL) {
        *os_out = *os_in;
        if (!os_out->fillStructureMap()) return false;
    }

    return true;
}

bool AbstractOSPRayStructure::getExtendsCallback(megamol::core::Call &call) {
    CallOSPRayStructure *os_in = dynamic_cast<CallOSPRayStructure*>(&call);
    CallOSPRayStructure *os_out = this->getStructureSlot.CallAs<CallOSPRayStructure>();

    if (os_in != NULL) {
        if (!this->getExtends(call)) {
            return false;
        }
        os_in->addExtend(this->extendContainer);
    }

    if (os_out != NULL) {
        *os_out = *os_in;
        if (!os_out->fillExtendMap()) return false;
    }

    return true;
}

