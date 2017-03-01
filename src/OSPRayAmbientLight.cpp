/*
* OSPRayAmbientLight.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayAmbientLight.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::ospray;


OSPRayAmbientLight::OSPRayAmbientLight(void) : AbstractOSPRayLight() { }

OSPRayAmbientLight::~OSPRayAmbientLight(void) { }

void OSPRayAmbientLight::readParams() {
    lightContainer.lightType = lightenum::AMBIENTLIGHT;
    auto lcolor = this->lightColor.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.lightColor.assign(lcolor, lcolor + 3);
    lightContainer.lightIntensity = this->lightIntensity.Param<core::param::FloatParam>()->Value();
}

bool OSPRayAmbientLight::InterfaceIsDirty() {
    if (this->AbstractIsDirty()) {
        return true;
    } else {
        return false;
    }
}