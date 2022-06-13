/*
 * SpotLight.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "mmstd/light/SpotLight.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::core::view::light;

void megamol::core::view::light::SpotLight::addLight(LightCollection& light_collection) {
    light_collection.add<SpotLightType>(std::static_pointer_cast<SpotLightType>(lightsource));
}

/*
 * megamol::core::view::light::SpotLight::SpotLight
 */
SpotLight::SpotLight(void)
        : AbstractLight()
        , position("Position", "")
        , direction("Direction", "")
        , openingAngle("openingAngle", "")
        , penumbraAngle("penumbraAngle", "")
        , radius("Radius", "") {

    // spot light
    lightsource = std::make_shared<SpotLightType>();

    this->position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    this->openingAngle << new core::param::FloatParam(0.0f);
    this->penumbraAngle << new core::param::FloatParam(0.0f);
    this->radius << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->position);
    this->MakeSlotAvailable(&this->direction);
    this->MakeSlotAvailable(&this->openingAngle);
    this->MakeSlotAvailable(&this->penumbraAngle);
    this->MakeSlotAvailable(&this->radius);
}

/*
 * megamol::core::view::light::SpotLight::~SpotLight
 */
SpotLight::~SpotLight(void) {
    this->Release();
}

/*
 * megamol::core::view::light::SpotLight::readParams
 */
void SpotLight::readParams() {
    auto light = std::static_pointer_cast<SpotLightType>(lightsource);

    light->colour = this->lightColor.Param<core::param::ColorParam>()->Value();
    light->intensity = this->lightIntensity.Param<core::param::FloatParam>()->Value();

    auto sl_pos = this->position.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    std::copy(sl_pos, sl_pos + 3, light->position.begin());
    auto sl_dir = this->direction.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    std::copy(sl_dir, sl_dir + 3, light->direction.begin());
    light->openingAngle = this->openingAngle.Param<core::param::FloatParam>()->Value();
    light->penumbraAngle = this->penumbraAngle.Param<core::param::FloatParam>()->Value();
    light->radius = this->radius.Param<core::param::FloatParam>()->Value();
}

/*
 * megamol::core::view::light::SpotLight::InterfaceIsDirty
 */
bool SpotLight::InterfaceIsDirty() {
    if (this->AbstractIsDirty() || this->position.IsDirty() || this->direction.IsDirty() ||
        this->openingAngle.IsDirty() || this->penumbraAngle.IsDirty() || this->radius.IsDirty()) {
        this->position.ResetDirty();
        this->direction.ResetDirty();
        this->openingAngle.ResetDirty();
        this->penumbraAngle.ResetDirty();
        this->radius.ResetDirty();
        return true;
    } else {
        return false;
    }
}
