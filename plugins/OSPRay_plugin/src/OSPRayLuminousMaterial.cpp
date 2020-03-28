/*
* OSPRayLuminousMaterial.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayLuminousMaterial.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::ospray;


OSPRayLuminousMaterial::OSPRayLuminousMaterial(void) :
    AbstractOSPRayMaterial(),
    // LUMINOUS
    lumColor("Color", "Color of the emitted light"),
    lumIntensity("Intensity", "Intensity of the emitted light"),
    lumTransparency("Transparency", "Transparency of the light source geometry") {

    this->lumColor << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 1.0f, 1.0f));
    this->lumIntensity << new core::param::FloatParam(1.0f);
    this->lumTransparency << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->lumColor);
    this->MakeSlotAvailable(&this->lumIntensity);
    this->MakeSlotAvailable(&this->lumTransparency);

}

OSPRayLuminousMaterial::~OSPRayLuminousMaterial(void) {
    this->Release();
}

void OSPRayLuminousMaterial::readParams() {
    materialContainer.materialType = materialTypeEnum::LUMINOUS;

    auto lumcolor = this->lumColor.Param<core::param::Vector3fParam>();
    materialContainer.lumColor = lumcolor->getArray();

    materialContainer.lumIntensity = this->lumIntensity.Param<core::param::FloatParam>()->Value();

    materialContainer.lumTransparency = this->lumTransparency.Param<core::param::FloatParam>()->Value();
}

bool OSPRayLuminousMaterial::InterfaceIsDirty() {
    if (
        this->lumColor.IsDirty() ||
        this->lumIntensity.IsDirty() ||
        this->lumTransparency.IsDirty() 
        ) {
        this->lumColor.ResetDirty();
        this->lumIntensity.ResetDirty();
        this->lumTransparency.ResetDirty();
        return true;
    } else {
        return false;
    }
}