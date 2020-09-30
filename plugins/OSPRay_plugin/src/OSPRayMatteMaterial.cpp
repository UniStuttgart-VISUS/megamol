/*
* OSPRayMatteMaterial.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayMatteMaterial.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::ospray;


OSPRayMatteMaterial::OSPRayMatteMaterial(void) :
    AbstractOSPRayMaterial(),
    // MATTE
    matteReflectance("Reflectance", "") {

    this->matteReflectance << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 1.0f, 1.0f));
    this->MakeSlotAvailable(&this->matteReflectance);
}

OSPRayMatteMaterial::~OSPRayMatteMaterial(void) {
    this->Release();
}

void OSPRayMatteMaterial::readParams() {
    materialContainer.materialType = materialTypeEnum::MATTE;

    auto reflect = this->matteReflectance.Param<core::param::Vector3fParam>();
    materialContainer.matteReflectance = reflect->getArray();
}

bool OSPRayMatteMaterial::InterfaceIsDirty() {
    if (
        this->matteReflectance.IsDirty()
        ) {
        this->matteReflectance.ResetDirty();
        return true;
    } else {
        return false;
    }
}