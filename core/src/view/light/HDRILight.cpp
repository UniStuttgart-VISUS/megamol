/*
 * HDRILight.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "mmcore/view/light/HDRILight.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::core::view::light;

void megamol::core::view::light::HDRILight::addLight(LightCollection& light_collection) {
    light_collection.add<HDRILightType>(std::static_pointer_cast<HDRILightType>(lightsource));
}

/*
 * megamol::core::view::light::HDRILight::HDRILight
 */
HDRILight::HDRILight(void) : AbstractLight(), up("up", ""), direction("Direction", ""), evnfile("EvironmentFile", "") {

    // HDRI light
    lightsource = std::make_shared<HDRILightType>();

    this->up << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    this->direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f));
    this->evnfile << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->up);
    this->MakeSlotAvailable(&this->direction);
    this->MakeSlotAvailable(&this->evnfile);
}

/*
 * megamol::core::view::light::HDRILight::~HDRILight
 */
HDRILight::~HDRILight(void) {
    this->Release();
}

/*
 * megamol::core::view::light::HDRILight::readParams
 */
void HDRILight::readParams() {
    auto light = std::static_pointer_cast<HDRILightType>(lightsource);

    light->colour = this->lightColor.Param<core::param::ColorParam>()->Value();
    light->intensity = this->lightIntensity.Param<core::param::FloatParam>()->Value();

    auto hdriup = this->up.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    std::copy(hdriup, hdriup + 3, light->up.begin());
    auto hdri_dir = this->direction.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    std::copy(hdri_dir, hdri_dir + 3, light->direction.begin());
    light->evnfile = this->evnfile.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str();
}

/*
 * megamol::core::view::light::HDRILight::InterfaceIsDirty
 */
bool HDRILight::InterfaceIsDirty() {
    if (this->AbstractIsDirty() || this->up.IsDirty() || this->direction.IsDirty() || this->evnfile.IsDirty()) {
        this->up.ResetDirty();
        this->direction.ResetDirty();
        this->evnfile.ResetDirty();
        return true;
    } else {
        return false;
    }
}
