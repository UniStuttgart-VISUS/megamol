/*
 * QuadLight.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/light/QuadLight.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/ColorParam.h"

using namespace megamol::core::view::light;

/*
 * megamol::core::view::light::QuadLight::QuadLight
 */
QuadLight::QuadLight(void)
    : AbstractLight()
    ,
    // quad light parameters
    ql_position("Position", "")
    , ql_edgeOne("Edge1", "")
    , ql_edgeTwo("Edge2", "") {

    // quad light
    this->ql_position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f));
    this->ql_edgeOne << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    this->ql_edgeTwo << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->ql_position);
    this->MakeSlotAvailable(&this->ql_edgeOne);
    this->MakeSlotAvailable(&this->ql_edgeTwo);
}

/*
 * megamol::core::view::light::QuadLight::~QuadLight
 */
QuadLight::~QuadLight(void) { this->Release(); }

/*
 * megamol::core::view::light::QuadLight::readParams
 */
void QuadLight::readParams() {
    lightContainer.lightType = lightenum::QUADLIGHT;
	lightContainer.lightColor = this->lightColor.Param<core::param::ColorParam>()->Value();
    lightContainer.lightIntensity = this->lightIntensity.Param<core::param::FloatParam>()->Value();

    auto ql_pos = this->ql_position.Param<core::param::Vector3fParam>()->Value().PeekComponents();
	std::copy(ql_pos, ql_pos + 3, lightContainer.ql_position.begin());
    auto ql_e1 = this->ql_edgeOne.Param<core::param::Vector3fParam>()->Value().PeekComponents();
	std::copy(ql_e1, ql_e1 + 3, lightContainer.ql_edgeOne.begin());
    auto ql_e2 = this->ql_edgeTwo.Param<core::param::Vector3fParam>()->Value().PeekComponents();
	std::copy(ql_e2, ql_e2 + 3, lightContainer.ql_edgeTwo.begin());
}

/*
 * megamol::core::view::light::QuadLight::InterfaceIsDirty
 */
bool QuadLight::InterfaceIsDirty() {
    if (this->AbstractIsDirty() || this->ql_position.IsDirty() || this->ql_edgeOne.IsDirty() ||
        this->ql_edgeTwo.IsDirty()) {
        this->ql_position.ResetDirty();
        this->ql_edgeOne.ResetDirty();
        this->ql_edgeTwo.ResetDirty();
        return true;
    } else {
        return false;
    }
}
