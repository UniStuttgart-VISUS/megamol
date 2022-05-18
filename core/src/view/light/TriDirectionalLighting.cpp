/*
 * TriDirectionalLighting.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "mmcore/view/light/TriDirectionalLighting.h"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "stdafx.h"

using namespace megamol::core::view::light;

void megamol::core::view::light::TriDirectionalLighting::addLight(LightCollection& light_collection) {
    light_collection.add<TriDirectionalLightType>(std::static_pointer_cast<TriDirectionalLightType>(lightsource));
}

/*
 * megamol::core::view::light::DistantLight::DistantLight
 */
TriDirectionalLighting::TriDirectionalLighting(void)
        : AbstractLight()
        , m_key_direction("KeyDirection", "Direction of the key light")
        , m_fill_direction("FillDirection", "Direction of the fill light")
        , m_back_direction("BackDirection", "Direction of the back light")
        , m_in_view_space("InViewSpace", "Locks the light directions to the camera") {

    // distant light
    lightsource = std::make_shared<TriDirectionalLightType>();

    this->m_key_direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(-0.2f, -0.2f, 1.0f));
    this->m_fill_direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f));
    this->m_back_direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, -1.0f));
    this->m_in_view_space << new core::param::BoolParam(1);
    this->MakeSlotAvailable(&this->m_key_direction);
    this->MakeSlotAvailable(&this->m_fill_direction);
    this->MakeSlotAvailable(&this->m_back_direction);
    this->MakeSlotAvailable(&this->m_in_view_space);

    this->m_key_direction.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Direction);
    this->m_fill_direction.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Direction);
    this->m_back_direction.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Direction);
}

/*
 * megamol::core::view::light::DistantLight::~DistantLight
 */
TriDirectionalLighting::~TriDirectionalLighting(void) {
    this->Release();
}

/*
 * megamol::core::view::light::DistantLight::readParams
 */
void TriDirectionalLighting::readParams() {
    auto light = std::static_pointer_cast<TriDirectionalLightType>(lightsource);

    // Read basic light parameters
    light->colour = this->lightColor.Param<core::param::ColorParam>()->Value();
    light->intensity = this->lightIntensity.Param<core::param::FloatParam>()->Value();

    // Read tri-directional lighting parameters
    light->in_view_space = this->m_in_view_space.Param<core::param::BoolParam>()->Value();
    if (light->in_view_space) {
        ++version; // force update every frame is eye direction is used
    }

    auto& key_dir = this->m_key_direction.Param<core::param::Vector3fParam>()->Value();
    glm::vec3 key_dir_normalized(key_dir.X(), key_dir.Y(), key_dir.Z());
    key_dir_normalized = glm::normalize(key_dir_normalized);
    std::copy(glm::value_ptr(key_dir_normalized), glm::value_ptr(key_dir_normalized) + 3, light->key_direction.begin());

    auto& fill_dir = this->m_fill_direction.Param<core::param::Vector3fParam>()->Value();
    glm::vec3 fill_dir_normalized(fill_dir.X(), fill_dir.Y(), fill_dir.Z());
    fill_dir_normalized = glm::normalize(fill_dir_normalized);
    std::copy(
        glm::value_ptr(fill_dir_normalized), glm::value_ptr(fill_dir_normalized) + 3, light->fill_direction.begin());

    auto& back_dir = this->m_back_direction.Param<core::param::Vector3fParam>()->Value();
    glm::vec3 back_dir_normalized(back_dir.X(), back_dir.Y(), back_dir.Z());
    back_dir_normalized = glm::normalize(back_dir_normalized);
    std::copy(
        glm::value_ptr(back_dir_normalized), glm::value_ptr(back_dir_normalized) + 3, light->back_direction.begin());
}

/*
 * megamol::core::view::light::DistantLight::InterfaceIsDirty
 */
bool TriDirectionalLighting::InterfaceIsDirty() {
    if (this->AbstractIsDirty() || this->m_key_direction.IsDirty() || this->m_fill_direction.IsDirty() ||
        this->m_back_direction.IsDirty() || this->m_in_view_space.IsDirty()) {
        this->m_key_direction.ResetDirty();
        this->m_fill_direction.ResetDirty();
        this->m_back_direction.ResetDirty();
        this->m_in_view_space.ResetDirty();
        return true;
    } else {
        return false;
    }
}
