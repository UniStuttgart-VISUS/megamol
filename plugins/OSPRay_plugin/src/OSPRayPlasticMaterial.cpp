/*
* OSPRayPlasticMaterial.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayPlasticMaterial.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::ospray;


OSPRayPlasticMaterial::OSPRayPlasticMaterial(void) :
    AbstractOSPRayMaterial(),
    // PLASTIC
    plasticPigmentColor("PigmentColor", ""),
    plasticEta("Eta", ""),
    plasticRoughness("Roughness", ""),
    plasticThickness("Thickness", "") {

    this->plasticPigmentColor << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.8f, 0.8f, 0.8f));
    this->plasticEta << new core::param::FloatParam(1.4f);
    this->plasticRoughness << new core::param::FloatParam(0.01f);
    this->plasticThickness << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->plasticEta);
    this->MakeSlotAvailable(&this->plasticPigmentColor);
    this->MakeSlotAvailable(&this->plasticRoughness);
    this->MakeSlotAvailable(&this->plasticThickness);

}

OSPRayPlasticMaterial::~OSPRayPlasticMaterial(void) {
    this->Release();
}

void OSPRayPlasticMaterial::readParams() {
    materialContainer.materialType = materialTypeEnum::PLASTIC;

    auto pcolor = this->plasticPigmentColor.Param<core::param::Vector3fParam>();
    materialContainer.plasticPigmentColor = pcolor->getArray();

    materialContainer.plasticEta = this->plasticEta.Param<core::param::FloatParam>()->Value();

    materialContainer.plasticRoughness = this->plasticRoughness.Param<core::param::FloatParam>()->Value();

    materialContainer.plasticThickness = this->plasticThickness.Param<core::param::FloatParam>()->Value();
}

bool OSPRayPlasticMaterial::InterfaceIsDirty() {
    if (
        this->plasticEta.IsDirty() ||
        this->plasticPigmentColor.IsDirty() ||
        this->plasticRoughness.IsDirty() ||
        this->plasticThickness.IsDirty()
        ) {
        this->plasticEta.ResetDirty();
        this->plasticPigmentColor.ResetDirty();
        this->plasticRoughness.ResetDirty();
        this->plasticThickness.ResetDirty();
        return true;
    } else {
        return false;
    }
}