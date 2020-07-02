/*
* OSPRayThinGlassMaterial.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayThinGlassMaterial.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::ospray;


OSPRayThinGlassMaterial::OSPRayThinGlassMaterial(void) :
    AbstractOSPRayMaterial(),
    // THINGLASS
    thinglassTransmission("Transmission", ""),
    thinglassEta("Eta", ""),
    thinglassThickness("Thickness", "") {

    this->thinglassTransmission << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 1.0f, 1.0f));
    this->thinglassEta << new core::param::FloatParam(1.5f);
    this->thinglassThickness << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->thinglassEta);
    this->MakeSlotAvailable(&this->thinglassThickness);
    this->MakeSlotAvailable(&this->thinglassTransmission);
}

OSPRayThinGlassMaterial::~OSPRayThinGlassMaterial(void) {
    this->Release();
}

void OSPRayThinGlassMaterial::readParams() {
    materialContainer.materialType = materialTypeEnum::THINGLASS;

    auto transmission = this->thinglassTransmission.Param<core::param::Vector3fParam>();
    materialContainer.thinglassTransmission = transmission->getArray();

    materialContainer.thinglassEta = this->thinglassEta.Param<core::param::FloatParam>()->Value();

    materialContainer.thinglassThickness = this->thinglassThickness.Param<core::param::FloatParam>()->Value();
}

bool OSPRayThinGlassMaterial::InterfaceIsDirty() {
    if (
        this->thinglassEta.IsDirty() ||
        this->thinglassThickness.IsDirty() ||
        this->thinglassTransmission.IsDirty()
        ) {
        this->thinglassEta.ResetDirty();
        this->thinglassThickness.ResetDirty();
        this->thinglassTransmission.ResetDirty();
        return true;
    } else {
        return false;
    }
}