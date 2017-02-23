/*
* OSPRayPointLight.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayPointLight.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::ospray;


OSPRayPointLight::OSPRayPointLight(void) :
    AbstractOSPRayLight(),
    // point light parameters
    pl_position("Light::PointLight::Position", ""),
    pl_radius("Light::PointLight::Radius", "")
{
    // point light
    this->pl_position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->pl_radius << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->pl_position);
    this->MakeSlotAvailable(&this->pl_radius);

}


void OSPRayPointLight::readParams() {
    lightContainer.lightType = lightenum::POINTLIGHT;
    auto lcolor = this->lightColor.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.lightColor.assign(lcolor, lcolor + 3);
    lightContainer.lightIntensity = this->lightIntensity.Param<core::param::FloatParam>()->Value();

    auto pl_pos = this->pl_position.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.pl_position.assign(pl_pos, pl_pos + 3);
    lightContainer.pl_radius = this->pl_radius.Param<core::param::FloatParam>()->Value();
}

bool OSPRayPointLight::InterfaceIsDirty() {
    if (this->AbstractIsDirty() ||
        this->pl_position.IsDirty() ||
        this->pl_radius.IsDirty() 
        ) {
        this->pl_position.ResetDirty();
        this->pl_radius.ResetDirty();
        return true;
    } else {
        return false;
    }
}