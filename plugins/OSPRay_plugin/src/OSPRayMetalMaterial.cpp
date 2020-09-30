/*
* OSPRayMetalMaterial.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayMetalMaterial.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::ospray;


OSPRayMetalMaterial::OSPRayMetalMaterial(void) :
    AbstractOSPRayMaterial(),
    // METAL
    metalReflectance("Reflectance", ""),
    metalEta("Eta", ""),
    metalK("K", ""),
    metalRoughness("Roughness", "") {

    this->metalReflectance << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 1.0f, 1.0f));
    this->metalEta << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.69700277f, 0.879832864f, 0.5301736f));
    this->metalK << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(9.30200672f, 6.27604008f, 4.89433956f));
    this->metalRoughness << new core::param::FloatParam(0.1f);

    this->MakeSlotAvailable(&this->metalEta);
    this->MakeSlotAvailable(&this->metalK);
    this->MakeSlotAvailable(&this->metalReflectance);
    this->MakeSlotAvailable(&this->metalRoughness);
}

OSPRayMetalMaterial::~OSPRayMetalMaterial(void) {
    this->Release();
}

void OSPRayMetalMaterial::readParams() {
    materialContainer.materialType = materialTypeEnum::METAL;

    auto reflect = this->metalReflectance.Param<core::param::Vector3fParam>();
    materialContainer.metalReflectance = reflect->getArray();

    auto eta = this->metalEta.Param<core::param::Vector3fParam>();
    materialContainer.metalEta = eta->getArray();

    auto k = this->metalK.Param<core::param::Vector3fParam>();
    materialContainer.metalK = k->getArray();

    materialContainer.metalRoughness = this->metalRoughness.Param<core::param::FloatParam>()->Value();
}

bool OSPRayMetalMaterial::InterfaceIsDirty() {
    if (
        this->metalEta.IsDirty() ||
        this->metalK.IsDirty() ||
        this->metalReflectance.IsDirty() ||
        this->metalRoughness.IsDirty() 
        ) {
        this->metalEta.ResetDirty();
        this->metalK.ResetDirty();
        this->metalReflectance.ResetDirty();
        this->metalRoughness.ResetDirty();
        return true;
    } else {
        return false;
    }
}