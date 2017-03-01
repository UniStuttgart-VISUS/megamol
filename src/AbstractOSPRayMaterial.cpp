/*
* AbstractOSPRayMaterial.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "AbstractOSPRayMaterial.h"
#include "CallOSPRayMaterial.h"
#include "vislib/sys/Log.h"



using namespace megamol::ospray;


AbstractOSPRayMaterial::AbstractOSPRayMaterial(void) :
    core::Module(),
    deployMaterialSlot("deployMaterialSlot", "Connects to an OSPRay geometry")  {

    this->deployMaterialSlot.SetCallback(CallOSPRayMaterial::ClassName(), CallOSPRayMaterial ::FunctionName(0), &AbstractOSPRayMaterial::getMaterialCallback);
    this->MakeSlotAvailable(&this->deployMaterialSlot);

    this->create();
}

AbstractOSPRayMaterial::~AbstractOSPRayMaterial(void) {
    this->release();
}

bool AbstractOSPRayMaterial::create() {
    this->materialContainer.isValid = true;
    return true;
}

void AbstractOSPRayMaterial ::release() {
    materialContainer.isValid = false;
}


bool AbstractOSPRayMaterial::getMaterialCallback(megamol::core::Call& call) {
    CallOSPRayMaterial *mc_in = dynamic_cast<CallOSPRayMaterial*>(&call);

    if (mc_in != NULL) {
        this->readParams();
        mc_in->setMaterialContainer(std::make_shared<OSPRayMaterialContainer>(std::move(this->materialContainer)));
    }

    return true;
}

