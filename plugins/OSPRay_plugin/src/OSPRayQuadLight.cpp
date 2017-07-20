/*
* OSPRayQuadLight.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayQuadLight.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::ospray;


OSPRayQuadLight::OSPRayQuadLight(void) :
    AbstractOSPRayLight(),
    // quad light parameters
    ql_position("Position", ""),
    ql_edgeOne("Edge1", ""),
    ql_edgeTwo("Edge2", "") {

    // quad light
    this->ql_position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f));
    this->ql_edgeOne << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    this->ql_edgeTwo << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->ql_position);
    this->MakeSlotAvailable(&this->ql_edgeOne);
    this->MakeSlotAvailable(&this->ql_edgeTwo);

}

OSPRayQuadLight::~OSPRayQuadLight(void) {
    this->Release();
}


void OSPRayQuadLight::readParams() {
    lightContainer.lightType = lightenum::QUADLIGHT;
    auto lcolor = this->lightColor.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.lightColor.assign(lcolor, lcolor + 3);
    lightContainer.lightIntensity = this->lightIntensity.Param<core::param::FloatParam>()->Value();

    auto ql_pos = this->ql_position.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.ql_position.assign(ql_pos, ql_pos + 3);
    auto ql_e1 = this->ql_edgeOne.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.ql_edgeOne.assign(ql_e1, ql_e1 + 3);
    auto ql_e2 = this->ql_edgeTwo.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.ql_edgeTwo.assign(ql_e2, ql_e2 + 3);
}

bool OSPRayQuadLight::InterfaceIsDirty() {
    if (this->AbstractIsDirty() ||
        this->ql_position.IsDirty() ||
        this->ql_edgeOne.IsDirty() ||
        this->ql_edgeTwo.IsDirty() 
        ) {
        this->ql_position.ResetDirty();
        this->ql_edgeOne.ResetDirty();
        this->ql_edgeTwo.ResetDirty();
        return true;
    } else {
        return false;
    }
}