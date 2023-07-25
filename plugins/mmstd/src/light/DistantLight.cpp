/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd/light/DistantLight.h"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::core::view::light;

void megamol::core::view::light::DistantLight::addLight(LightCollection& light_collection) {
    light_collection.add<DistantLightType>(std::static_pointer_cast<DistantLightType>(lightsource));
}

/*
 * megamol::core::view::light::DistantLight::DistantLight
 */
DistantLight::DistantLight()
        : AbstractLight()
        , direction("Direction", "Direction of the Light")
        , angularDiameter("AngularDiameter", "If greater than zero results in soft shadows")
        , eye_direction("EyeDirection", "Sets the light direction as view direction") {

    // distant light
    lightsource = std::make_shared<DistantLightType>();

    this->angularDiameter << new core::param::FloatParam(0.0f);
    this->direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(-0.25f, -0.5f, -0.75f));
    this->eye_direction << new core::param::BoolParam(0);
    this->MakeSlotAvailable(&this->direction);
    this->MakeSlotAvailable(&this->angularDiameter);
    this->MakeSlotAvailable(&this->eye_direction);

    this->direction.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::Direction);
}

/*
 * megamol::core::view::light::DistantLight::~DistantLight
 */
DistantLight::~DistantLight() {
    this->Release();
}

/*
 * megamol::core::view::light::DistantLight::readParams
 */
void DistantLight::readParams() {
    auto light = std::static_pointer_cast<DistantLightType>(lightsource);

    light->colour = this->lightColor.Param<core::param::ColorParam>()->Value();
    light->intensity = this->lightIntensity.Param<core::param::FloatParam>()->Value();

    light->eye_direction = this->eye_direction.Param<core::param::BoolParam>()->Value();
    if (light->eye_direction) {
        ++version;
    } // force update every frame is eye direction is used
    auto& dl_dir = this->direction.Param<core::param::Vector3fParam>()->Value();
    glm::vec3 dl_dir_normalized(dl_dir.X(), dl_dir.Y(), dl_dir.Z());
    dl_dir_normalized = glm::normalize(dl_dir_normalized);
    std::copy(glm::value_ptr(dl_dir_normalized), glm::value_ptr(dl_dir_normalized) + 3, light->direction.begin());
    light->angularDiameter = this->angularDiameter.Param<core::param::FloatParam>()->Value();
}

/*
 * megamol::core::view::light::DistantLight::InterfaceIsDirty
 */
bool DistantLight::InterfaceIsDirty() {
    if (this->AbstractIsDirty() || this->angularDiameter.IsDirty() || this->direction.IsDirty() ||
        this->eye_direction.IsDirty()) {
        this->angularDiameter.ResetDirty();
        this->direction.ResetDirty();
        this->eye_direction.ResetDirty();
        return true;
    } else {
        return false;
    }
}
