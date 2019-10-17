/*
 * HDRILight.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/light/HDRILight.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/ColorParam.h"

using namespace megamol::core::view::light;

/*
 * megamol::core::view::light::HDRILight::HDRILight
 */
HDRILight::HDRILight(void)
    : AbstractLight()
    ,
    // hdri light parameteres
    hdri_up("up", "")
    , hdri_direction("Direction", "")
    , hdri_evnfile("EvironmentFile", "") {

    // HDRI light
    this->hdri_up << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    this->hdri_direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f));
    this->hdri_evnfile << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->hdri_up);
    this->MakeSlotAvailable(&this->hdri_direction);
    this->MakeSlotAvailable(&this->hdri_evnfile);
}

/*
 * megamol::core::view::light::HDRILight::~HDRILight
 */
HDRILight::~HDRILight(void) { this->Release(); }

/*
 * megamol::core::view::light::HDRILight::readParams
 */
void HDRILight::readParams() {
    lightContainer.lightType = lightenum::HDRILIGHT;
	lightContainer.lightColor = this->lightColor.Param<core::param::ColorParam>()->Value();
    lightContainer.lightIntensity = this->lightIntensity.Param<core::param::FloatParam>()->Value();

    auto hdriup = this->hdri_up.Param<core::param::Vector3fParam>()->Value().PeekComponents();
	std::copy(hdriup, hdriup + 3, lightContainer.hdri_up.begin());
    auto hdri_dir = this->hdri_direction.Param<core::param::Vector3fParam>()->Value().PeekComponents();
	std::copy(hdri_dir, hdri_dir + 3, lightContainer.hdri_direction.begin());
    lightContainer.hdri_evnfile = this->hdri_evnfile.Param<core::param::FilePathParam>()->Value();
}

/*
 * megamol::core::view::light::HDRILight::InterfaceIsDirty
 */
bool HDRILight::InterfaceIsDirty() {
    if (this->AbstractIsDirty() || this->hdri_up.IsDirty() || this->hdri_direction.IsDirty() ||
        this->hdri_evnfile.IsDirty()) {
        this->hdri_up.ResetDirty();
        this->hdri_direction.ResetDirty();
        this->hdri_evnfile.ResetDirty();
        return true;
    } else {
        return false;
    }
}
