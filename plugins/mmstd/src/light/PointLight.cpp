/*
 * PointLight.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "mmstd/light/PointLight.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::core::view::light;

void megamol::core::view::light::PointLight::addLight(LightCollection& light_collection) {
    light_collection.add<PointLightType>(std::static_pointer_cast<PointLightType>(lightsource));
}

/*
 * megamol::core::view::light::PointLight::PointLight
 */
PointLight::PointLight(void) : AbstractLight(), position("Position", ""), radius("Radius", "") {

    // point light
    lightsource = std::make_shared<PointLightType>();

    this->position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->radius << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->position);
    this->MakeSlotAvailable(&this->radius);
}

/*
 * megamol::core::view::light::PointLight::~PointLight
 */
PointLight::~PointLight(void) {
    this->Release();
}

/*
 * megamol::core::view::light::PointLight::readParams
 */
void PointLight::readParams() {
    auto light = std::static_pointer_cast<PointLightType>(lightsource);

    light->colour = this->lightColor.Param<core::param::ColorParam>()->Value();
    light->intensity = this->lightIntensity.Param<core::param::FloatParam>()->Value();

    auto pl_pos = this->position.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    std::copy(pl_pos, pl_pos + 3, light->position.begin());
    light->radius = this->radius.Param<core::param::FloatParam>()->Value();
}

/*
 * megamol::core::view::light::PointLight::InterfaceIsDirty
 */
bool PointLight::InterfaceIsDirty() {
    if (this->AbstractIsDirty() || this->position.IsDirty() || this->radius.IsDirty()) {
        this->position.ResetDirty();
        this->radius.ResetDirty();
        return true;
    } else {
        return false;
    }
}
