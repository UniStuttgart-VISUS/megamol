/*
 * AbstractLight.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/light/AbstractLight.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/ColorParam.h"
#include "vislib/sys/Log.h"

using namespace megamol::core::view::light;

/*
 * megamol::core::view::light::AbstractLight::AbstractLight
 */
AbstractLight::AbstractLight(void)
    : core::Module()
    , lightContainer()
    , getLightSlot("getLightSlot", "Connects to another light")
    , deployLightSlot("deployLightSlot", "Connects to a renderer module or another light")
    // General light parameters
    , lightColor("Color", "Sets the color of the Light")
    , lightIntensity("Intensity", "Intensity of the Light") {

    this->lightContainer.isValid = true;

    this->getLightSlot.SetCompatibleCall<CallLightDescription>();
    this->MakeSlotAvailable(&this->getLightSlot);

    this->deployLightSlot.SetCallback(
        CallLight::ClassName(), CallLight::FunctionName(0), &AbstractLight::getLightCallback);
    this->MakeSlotAvailable(&this->deployLightSlot);

    // general light
    this->lightColor << new core::param::ColorParam(0.8f, 0.8f, 0.8f, 1.0f);
    this->lightIntensity << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->lightIntensity);
    this->MakeSlotAvailable(&this->lightColor);
}

/*
 * megamol::core::view::light::AbstractLight::~AbstractLight
 */
AbstractLight::~AbstractLight(void) {
    lightContainer.isValid = false;
    this->Release();
}

/*
 * megamol::core::view::light::AbstractLight::create
 */
bool AbstractLight::create() {
    // intentionally empty
    return true;
}

/*
 * megamol::core::view::light::AbstractLight::release
 */
void AbstractLight::release() {
    // intentionally empty
}

/*
 * megamol::core::view::light::AbstractLight::getLightCallback
 */
bool AbstractLight::getLightCallback(megamol::core::Call& call) {
    CallLight* lc_in = dynamic_cast<CallLight*>(&call);
    CallLight* lc_out = this->getLightSlot.CallAs<CallLight>();

    this->lightContainer.dataChanged = false;
    if (this->InterfaceIsDirty()) {
        this->lightContainer.dataChanged = true;
    }
    if (lc_in != NULL) {
        this->readParams();
        lc_in->addLight(lightContainer);
    }

    if (lc_out != NULL) {
        *lc_out = *lc_in;
        // lc_out->fillLightMap();
        if (!(*lc_out)(0)) return false;
    }

    return true;
}

/*
 * megamol::core::view::light::AbstractLight::AbstractIsDirty
 */
bool AbstractLight::AbstractIsDirty() {
    if (this->lightIntensity.IsDirty() || this->lightColor.IsDirty()) {
        this->lightIntensity.ResetDirty();
        this->lightColor.ResetDirty();
        return true;
    } else {
        return false;
    }
}
