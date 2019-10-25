/*
 * PointLight.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/light/PointLight.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/ColorParam.h"

using namespace megamol::core::view::light;

/*
 * megamol::core::view::light::PointLight::PointLight
 */
PointLight::PointLight(void)
    : AbstractLight()
    ,
    // point light parameters
    pl_position("Position", "")
    , pl_radius("Radius", "") {
    // point light
    this->pl_position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->pl_radius << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->pl_position);
    this->MakeSlotAvailable(&this->pl_radius);
}

/*
 * megamol::core::view::light::PointLight::~PointLight
 */
PointLight::~PointLight(void) { this->Release(); }

/*
 * megamol::core::view::light::PointLight::readParams
 */
void PointLight::readParams() {
    lightContainer.lightType = lightenum::POINTLIGHT;
	lightContainer.lightColor = this->lightColor.Param<core::param::ColorParam>()->Value();
    lightContainer.lightIntensity = this->lightIntensity.Param<core::param::FloatParam>()->Value();

    auto pl_pos = this->pl_position.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    std::copy(pl_pos, pl_pos+3, lightContainer.pl_position.begin());
    lightContainer.pl_radius = this->pl_radius.Param<core::param::FloatParam>()->Value();
}

/*
 * megamol::core::view::light::PointLight::InterfaceIsDirty
 */
bool PointLight::InterfaceIsDirty() {
    if (this->AbstractIsDirty() || this->pl_position.IsDirty() || this->pl_radius.IsDirty()) {
        this->pl_position.ResetDirty();
        this->pl_radius.ResetDirty();
        return true;
    } else {
        return false;
    }
}
