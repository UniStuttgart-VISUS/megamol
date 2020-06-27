/*
* OSPRayVelvetMaterial.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayVelvetMaterial.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::ospray;


OSPRayVelvetMaterial::OSPRayVelvetMaterial(void) :
    AbstractOSPRayMaterial(),
    // VELVET
    velvetReflectance("Reflectance", "Reflectance"),
    velvetBackScattering("BackScattering", "BackScattering"),
    velvetHorizonScatteringColor("HorizonScatteringColor", "Scattering color"),
    velvetHorizonScatteringFallOff("HorizonScatteringFallOff", "Scattering fall off") {

    this->velvetReflectance << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.4f, 0.0f, 0.0f));
    this->velvetHorizonScatteringColor << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.75f, 0.1f, 0.1f));
    this->velvetBackScattering << new core::param::FloatParam(0.5f);
    this->velvetHorizonScatteringFallOff << new core::param::FloatParam(10.0f);
    this->MakeSlotAvailable(&this->velvetBackScattering);
    this->MakeSlotAvailable(&this->velvetHorizonScatteringColor);
    this->MakeSlotAvailable(&this->velvetHorizonScatteringFallOff);
    this->MakeSlotAvailable(&this->velvetReflectance);
}

OSPRayVelvetMaterial::~OSPRayVelvetMaterial(void) {
    this->Release();
}

void OSPRayVelvetMaterial::readParams() {
    materialContainer.materialType = materialTypeEnum::VELVET;

    auto reflect = this->velvetReflectance.Param<core::param::Vector3fParam>();
    materialContainer.velvetReflectance = reflect->getArray();

    auto color = this->velvetHorizonScatteringColor.Param<core::param::Vector3fParam>();
    materialContainer.velvetHorizonScatteringColor = color->getArray();

    materialContainer.velvetBackScattering = this->velvetBackScattering.Param<core::param::FloatParam>()->Value();

    materialContainer.velvetHorizonScatteringFallOff = this->velvetHorizonScatteringFallOff.Param<core::param::FloatParam>()->Value();
}

bool OSPRayVelvetMaterial::InterfaceIsDirty() {
    if (
        this->velvetBackScattering.IsDirty() ||
        this->velvetHorizonScatteringColor.IsDirty() ||
        this->velvetHorizonScatteringFallOff.IsDirty() ||
        this->velvetReflectance.IsDirty() 
        ) {
        this->velvetBackScattering.ResetDirty();
        this->velvetHorizonScatteringColor.ResetDirty();
        this->velvetHorizonScatteringFallOff.ResetDirty();
        this->velvetReflectance.ResetDirty();
        return true;
    } else {
        return false;
    }
}