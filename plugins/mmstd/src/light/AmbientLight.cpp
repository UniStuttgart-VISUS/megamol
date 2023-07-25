/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd/light/AmbientLight.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FloatParam.h"

using namespace megamol::core::view::light;

void megamol::core::view::light::AmbientLight::addLight(LightCollection& light_collection) {
    light_collection.add<AmbientLightType>(std::static_pointer_cast<AmbientLightType>(lightsource));
}

/*
 * megamol::core::view::light::AmbientLight::AmbientLight
 */
AmbientLight::AmbientLight() : AbstractLight() {
    lightsource = std::make_shared<AmbientLightType>();
}

/*
 * megamol::core::view::light::AmbientLight::AmbientLight
 */
AmbientLight::~AmbientLight() {
    this->Release();
}

/*
 * megamol::core::view::light::AmbientLight::readParams
 */
void AmbientLight::readParams() {
    auto light = std::static_pointer_cast<AmbientLightType>(lightsource);

    light->colour = this->lightColor.Param<core::param::ColorParam>()->Value();
    light->intensity = this->lightIntensity.Param<core::param::FloatParam>()->Value();
}

/*
 * megamol::core::view::light::AmbientLight::InterfaceIsDirty
 */
bool AmbientLight::InterfaceIsDirty() {
    if (this->AbstractIsDirty()) {
        return true;
    } else {
        return false;
    }
}
