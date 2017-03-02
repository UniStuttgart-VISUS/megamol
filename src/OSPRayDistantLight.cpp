/*
* OSPRayDistantLight.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayDistantLight.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::ospray;


OSPRayDistantLight::OSPRayDistantLight(void) :
    AbstractOSPRayLight(),
    // Distant light parameters
    dl_direction("Direction", "Direction of the Light"),
    dl_angularDiameter("AngularDiameter", "If greater than zero results in soft shadows"),
    dl_eye_direction("EyeDirection", "Sets the light direction as view direction") {

    // distant light
    this->dl_angularDiameter << new core::param::FloatParam(0.0f);
    this->dl_direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, -1.0f, 0.0f));
    this->dl_eye_direction << new core::param::BoolParam(0);
    this->MakeSlotAvailable(&this->dl_direction);
    this->MakeSlotAvailable(&this->dl_angularDiameter);
    this->MakeSlotAvailable(&this->dl_eye_direction);

}

OSPRayDistantLight::~OSPRayDistantLight(void) {}


void OSPRayDistantLight::readParams() {
    lightContainer.lightType = lightenum::DISTANTLIGHT;
    auto lcolor = this->lightColor.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.lightColor.assign(lcolor, lcolor + 3);
    lightContainer.lightIntensity = this->lightIntensity.Param<core::param::FloatParam>()->Value();
    lightContainer.dl_eye_direction = this->dl_eye_direction.Param<core::param::BoolParam>()->Value();
    auto dl_dir = this->dl_direction.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.dl_direction.assign(dl_dir, dl_dir + 3);
    lightContainer.dl_angularDiameter = this->dl_angularDiameter.Param<core::param::FloatParam>()->Value();
}

bool OSPRayDistantLight::InterfaceIsDirty() {
    if (this->AbstractIsDirty() ||
        this->dl_angularDiameter.IsDirty() ||
        this->dl_direction.IsDirty() ||
        this->dl_eye_direction.IsDirty()
        ) {
        this->dl_angularDiameter.ResetDirty();
        this->dl_direction.ResetDirty();
        this->dl_eye_direction.ResetDirty();
        return true;
    } else {
        return false;
    }
}