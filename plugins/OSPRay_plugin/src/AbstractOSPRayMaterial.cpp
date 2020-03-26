/*
* AbstractOSPRayMaterial.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "AbstractOSPRayMaterial.h"
#include "OSPRay_plugin/CallOSPRayMaterial.h"
#include "vislib/sys/Log.h"



using namespace megamol::ospray;


AbstractOSPRayMaterial::AbstractOSPRayMaterial(void) :
    core::Module(),
    deployMaterialSlot("deployMaterialSlot", "Connects to an OSPRay geometry")
{
    this->materialContainer.isValid = true;

    this->deployMaterialSlot.SetCallback(CallOSPRayMaterial::ClassName(), CallOSPRayMaterial ::FunctionName(0), &AbstractOSPRayMaterial::getMaterialCallback);
    this->MakeSlotAvailable(&this->deployMaterialSlot);

}

AbstractOSPRayMaterial::~AbstractOSPRayMaterial(void) {
    materialContainer.isValid = false;
    this->Release();
}

bool AbstractOSPRayMaterial::create() {
    return true;
}

void AbstractOSPRayMaterial ::release() {

}


bool AbstractOSPRayMaterial::getMaterialCallback(megamol::core::Call& call) {
    CallOSPRayMaterial *mc_in = dynamic_cast<CallOSPRayMaterial*>(&call);

    if (mc_in != NULL) {
        this->readParams();
        mc_in->setMaterialContainer(std::make_shared<OSPRayMaterialContainer>(std::move(this->materialContainer)));
    }

    if (this->InterfaceIsDirty()) {
        mc_in->setDirty();
    }

    return true;
}

