/*
 * OSPRayThinGlassMaterial.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "OSPRayThinGlassMaterial.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::ospray;


OSPRayThinGlassMaterial::OSPRayThinGlassMaterial()
        : AbstractOSPRayMaterial()
        ,
        // THINGLASS
        thinglassTransmission("Transmission", "")
        , thinglassEta("Eta", "")
        , thinglassThickness("Thickness", "") {

    this->thinglassTransmission << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 1.0f, 1.0f));
    this->thinglassEta << new core::param::FloatParam(1.5f);
    this->thinglassThickness << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->thinglassEta);
    this->MakeSlotAvailable(&this->thinglassThickness);
    this->MakeSlotAvailable(&this->thinglassTransmission);
}

OSPRayThinGlassMaterial::~OSPRayThinGlassMaterial() {
    this->Release();
}

void OSPRayThinGlassMaterial::readParams() {
    materialContainer.materialType = materialTypeEnum::THINGLASS;

    thinglassMaterial tgm;

    auto transmission = this->thinglassTransmission.Param<core::param::Vector3fParam>();
    tgm.thinglassTransmission = transmission->getArray();

    tgm.thinglassEta = this->thinglassEta.Param<core::param::FloatParam>()->Value();

    tgm.thinglassThickness = this->thinglassThickness.Param<core::param::FloatParam>()->Value();

    materialContainer.material = tgm;
}

bool OSPRayThinGlassMaterial::InterfaceIsDirty() {
    if (this->thinglassEta.IsDirty() || this->thinglassThickness.IsDirty() || this->thinglassTransmission.IsDirty()) {
        this->thinglassEta.ResetDirty();
        this->thinglassThickness.ResetDirty();
        this->thinglassTransmission.ResetDirty();
        return true;
    } else {
        return false;
    }
}
