/*
 * AbstractOSPRayStructure.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "mmospray/AbstractOSPRayStructure.h"

#include "glm/glm.hpp"
#include "mmcore/view/CallClipPlane.h"


namespace megamol {
namespace ospray {


AbstractOSPRayStructure::AbstractOSPRayStructure()
        : Module()
        , deployStructureSlot("deployStructureSlot", "Connects to the OSPRayRenderer or another OSPRayStructure")
        , getStructureSlot("getStructureSlot", "Connects to the another OSPRayStructure")
        , getMaterialSlot("getMaterialSlot", "Connects to an OSPRayMaterial")
        , getTransformationSlot("getTransformationSlot", "Connects to an OSPRayTransform")
        , readFlagsSlot("readFlags", "")
        , writeFlagsSlot("writeFlags", "")
        , getClipplaneSlot("getClipPlaneSlot", "Connects to a Clipping plane slot") {

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

    this->getClipplaneSlot.SetCompatibleCall<core::view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipplaneSlot);

    readFlagsSlot.SetCompatibleCall<core::FlagCallRead_CPUDescription>();
    MakeSlotAvailable(&this->readFlagsSlot);
    writeFlagsSlot.SetCompatibleCall<core::FlagCallWrite_CPUDescription>();
    MakeSlotAvailable(&this->writeFlagsSlot);

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
    CallOSPRayStructure* os_in = dynamic_cast<CallOSPRayStructure*>(&call);
    CallOSPRayStructure* os_out = this->getStructureSlot.CallAs<CallOSPRayStructure>();

    if (os_in != NULL) {
        if (!this->readData(call))
            return false;
        os_in->addStructure(this->structureContainer);
    }

    if (os_out != NULL) {
        *os_out = *os_in;
        if (!os_out->fillStructureMap())
            return false;
    }

    return true;
}

bool AbstractOSPRayStructure::getExtendsCallback(megamol::core::Call& call) {
    CallOSPRayStructure* os_in = dynamic_cast<CallOSPRayStructure*>(&call);
    CallOSPRayStructure* os_out = this->getStructureSlot.CallAs<CallOSPRayStructure>();

    if (os_in != NULL) {
        if (!this->getExtends(call)) {
            return false;
        }
        os_in->addExtend(this->extendContainer);
    }

    if (os_out != NULL) {
        *os_out = *os_in;
        if (!os_out->fillExtendMap())
            return false;
    }

    return true;
}

void AbstractOSPRayStructure::processMaterial() {
    CallOSPRayMaterial* cm = this->getMaterialSlot.CallAs<CallOSPRayMaterial>();
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

void AbstractOSPRayStructure::processClippingPlane() {
    auto ccp = this->getClipplaneSlot.CallAs<core::view::CallClipPlane>();

    if ((ccp != nullptr) && (*ccp)()) {
        this->structureContainer.clippingPlane.isValid = true;
        glm::vec3 normal = {
            ccp->GetPlane().Normal().GetX(), ccp->GetPlane().Normal().GetY(), ccp->GetPlane().Normal().GetZ()};
        float d = ccp->GetPlane().D();
        //glm::vec3 point = {
        //    ccp->GetPlane().Point().GetX(), ccp->GetPlane().Point().GetY(), ccp->GetPlane().Point().GetZ()};
        //float d = glm::dot(point, normal);
        ClippingPlane& cp = this->structureContainer.clippingPlane;
        if (cp.coeff[0] != normal.x || cp.coeff[1] != normal.y || cp.coeff[2] != normal.z || cp.coeff[3] != d) {
            this->structureContainer.clippingPlaneChanged = true;
            cp.coeff[0] = normal.x;
            cp.coeff[1] = normal.y;
            cp.coeff[2] = normal.z;
            cp.coeff[3] = d;
        } else {
            this->structureContainer.clippingPlaneChanged = false;
        }
    }
}

} // namespace ospray
} // namespace megamol
