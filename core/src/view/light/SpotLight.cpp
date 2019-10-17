/*
 * SpotLight.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/light/SpotLight.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/ColorParam.h"

using namespace megamol::core::view::light;

/*
 * megamol::core::view::light::SpotLight::SpotLight
 */
SpotLight::SpotLight(void)
    : AbstractLight()
    ,
    // Distant light parameters
    // spot light parameters
    sl_position("Position", "")
    , sl_direction("Direction", "")
    , sl_openingAngle("openingAngle", "")
    , sl_penumbraAngle("penumbraAngle", "")
    , sl_radius("Radius", "") {

    // spot light
    this->sl_position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->sl_direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    this->sl_openingAngle << new core::param::FloatParam(0.0f);
    this->sl_penumbraAngle << new core::param::FloatParam(0.0f);
    this->sl_radius << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->sl_position);
    this->MakeSlotAvailable(&this->sl_direction);
    this->MakeSlotAvailable(&this->sl_openingAngle);
    this->MakeSlotAvailable(&this->sl_penumbraAngle);
    this->MakeSlotAvailable(&this->sl_radius);
}

/*
 * megamol::core::view::light::SpotLight::~SpotLight
 */
SpotLight::~SpotLight(void) { this->Release(); }

/*
 * megamol::core::view::light::SpotLight::readParams
 */
void SpotLight::readParams() {
    lightContainer.lightType = lightenum::SPOTLIGHT;
	lightContainer.lightColor = this->lightColor.Param<core::param::ColorParam>()->Value();
    lightContainer.lightIntensity = this->lightIntensity.Param<core::param::FloatParam>()->Value();

    auto sl_pos = this->sl_position.Param<core::param::Vector3fParam>()->Value().PeekComponents();
	std::copy(sl_pos, sl_pos + 3, lightContainer.sl_position.begin());
    auto sl_dir = this->sl_direction.Param<core::param::Vector3fParam>()->Value().PeekComponents();
	std::copy(sl_dir, sl_dir + 3, lightContainer.sl_direction.begin());
    lightContainer.sl_openingAngle = this->sl_openingAngle.Param<core::param::FloatParam>()->Value();
    lightContainer.sl_penumbraAngle = this->sl_penumbraAngle.Param<core::param::FloatParam>()->Value();
    lightContainer.sl_radius = this->sl_radius.Param<core::param::FloatParam>()->Value();
}

/*
 * megamol::core::view::light::SpotLight::InterfaceIsDirty
 */
bool SpotLight::InterfaceIsDirty() {
    if (this->AbstractIsDirty() || this->sl_position.IsDirty() || this->sl_direction.IsDirty() ||
        this->sl_openingAngle.IsDirty() || this->sl_penumbraAngle.IsDirty() || this->sl_radius.IsDirty()) {
        this->sl_position.ResetDirty();
        this->sl_direction.ResetDirty();
        this->sl_openingAngle.ResetDirty();
        this->sl_penumbraAngle.ResetDirty();
        this->sl_radius.ResetDirty();
        return true;
    } else {
        return false;
    }
}
