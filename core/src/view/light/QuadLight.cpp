/*
 * QuadLight.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "mmcore/view/light/QuadLight.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "stdafx.h"

using namespace megamol::core::view::light;

void megamol::core::view::light::QuadLight::addLight(LightCollection& light_collection) {
    light_collection.add<QuadLightType>(std::static_pointer_cast<QuadLightType>(lightsource));
}

/*
 * megamol::core::view::light::QuadLight::QuadLight
 */
QuadLight::QuadLight(void) : AbstractLight(), position("Position", ""), edgeOne("Edge1", ""), edgeTwo("Edge2", "") {

    // quad light
    lightsource = std::make_shared<QuadLightType>();

    this->position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f));
    this->edgeOne << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    this->edgeTwo << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->position);
    this->MakeSlotAvailable(&this->edgeOne);
    this->MakeSlotAvailable(&this->edgeTwo);
}

/*
 * megamol::core::view::light::QuadLight::~QuadLight
 */
QuadLight::~QuadLight(void) {
    this->Release();
}

/*
 * megamol::core::view::light::QuadLight::readParams
 */
void QuadLight::readParams() {
    auto light = std::static_pointer_cast<QuadLightType>(lightsource);

    light->colour = this->lightColor.Param<core::param::ColorParam>()->Value();
    light->intensity = this->lightIntensity.Param<core::param::FloatParam>()->Value();

    auto ql_pos = this->position.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    std::copy(ql_pos, ql_pos + 3, light->position.begin());
    auto ql_e1 = this->edgeOne.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    std::copy(ql_e1, ql_e1 + 3, light->edgeOne.begin());
    auto ql_e2 = this->edgeTwo.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    std::copy(ql_e2, ql_e2 + 3, light->edgeTwo.begin());
}

/*
 * megamol::core::view::light::QuadLight::InterfaceIsDirty
 */
bool QuadLight::InterfaceIsDirty() {
    if (this->AbstractIsDirty() || this->position.IsDirty() || this->edgeOne.IsDirty() || this->edgeTwo.IsDirty()) {
        this->position.ResetDirty();
        this->edgeOne.ResetDirty();
        this->edgeTwo.ResetDirty();
        return true;
    } else {
        return false;
    }
}
