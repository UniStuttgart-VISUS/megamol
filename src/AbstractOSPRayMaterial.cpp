/*
* AbstractOSPRayMaterial.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "AbstractOSPRayMaterial.h"
#include "vislib/sys/Log.h"



using namespace megamol::ospray;


AbstractOSPRayMaterial::AbstractOSPRayMaterial(void) :
    core::Module(),
    getMaterialSlot("getMaterialSlot", "Connects to the OSPRayRenderer or another OSPRayLight")  { }

AbstractOSPRayMaterial::~AbstractOSPRayMaterial(void) {
    AbstractOSPRayMaterial::release();
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
        if (InterfaceIsDirty()) {
            this->readParams();
            mc_in->setMaterialContainer(&(this->materialContainer));
        }
    }

    return true;
}

