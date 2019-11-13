/*
* AbstractOSPRayStructure.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRay_plugin/AbstractOSPRayStructure.h"

namespace megamol {
namespace ospray {


AbstractOSPRayStructure::AbstractOSPRayStructure()
    : Module()
    , deployStructureSlot("deployStructureSlot", "Connects to the OSPRayRenderer or another OSPRayStructure")
    , getStructureSlot("getStructureSlot", "Connects to the another OSPRayStructure")
    , getMaterialSlot("getMaterialSlot", "Connects to an OSPRayMaterial") 
    , getTransformationSlot("getTransformationSlot", "Connects to an OSPRayTransform") {

    this->deployStructureSlot.SetCallback(CallOSPRayStructure::ClassName(), CallOSPRayStructure::FunctionName(0),
        &AbstractOSPRayStructure::getStructureCallback);
    this->deployStructureSlot.SetCallback(CallOSPRayStructure::ClassName(), CallOSPRayStructure::FunctionName(1),
        &AbstractOSPRayStructure::getExtendsCallback);
    this->MakeSlotAvailable(&this->deployStructureSlot);

    this->getStructureSlot.SetCompatibleCall<CallOSPRayStructureDescription>();
    this->MakeSlotAvailable(&this->getStructureSlot);

    this->getMaterialSlot.SetCompatibleCall<CallOSPRayMaterialDescription>();
    this->MakeSlotAvailable(&this->getMaterialSlot);

    this->getTransformationSlot.SetCompatibleCall<CallOSPRayTransformationDescription>();
    this->MakeSlotAvailable(&this->getTransformationSlot);

    this->structureContainer.isValid = true;
    this->time = -1.0f;
}

AbstractOSPRayStructure::~AbstractOSPRayStructure() {
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

void AbstractOSPRayStructure::processMaterial() {
    CallOSPRayMaterial *cm = this->getMaterialSlot.CallAs<CallOSPRayMaterial>();
    if (cm != NULL) {
        this->structureContainer.materialChanged = false;
        if (cm->InterfaceIsDirty()) {
            this->structureContainer.materialChanged = true;
        }
        auto gmp = cm->getMaterialParameter();
        if (gmp->isValid) {
            this->structureContainer.materialContainer = cm->getMaterialParameter();
        }
    } else {
        this->structureContainer.materialChanged = false;
        this->structureContainer.materialContainer = nullptr;
    }
}

void AbstractOSPRayStructure::processTransformation() {
    CallOSPRayTransformation* ct = this->getTransformationSlot.CallAs<CallOSPRayTransformation>();
    if (ct != NULL) {
        this->structureContainer.transformationChanged = false;
        if (ct->InterfaceIsDirty()) {
            this->structureContainer.transformationChanged = true;
        }
        auto gmp = ct->getTransformationParameter();
        if (gmp->isValid) {
            this->structureContainer.transformationContainer = ct->getTransformationParameter();
        }
    } else {
        this->structureContainer.transformationChanged = false;
        this->structureContainer.transformationContainer = nullptr;
    }
}

} // namespace ospray
} // namespace megamol