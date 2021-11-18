/*
 * AbstractLight.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "mmcore/view/light/AbstractLight.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"

using namespace megamol::core::view::light;

/*
 * megamol::core::view::light::AbstractLight::AbstractLight
 */
AbstractLight::AbstractLight(void)
        : core::Module()
        , version(0)
        , lightsource(nullptr)
        , rhs_connected(false)
        , getLightSlot("getLightSlot", "Connects to another light")
        , deployLightSlot("deployLightSlot", "Connects to a renderer module or another light")
        // General light parameters
        , lightColor("Color", "Sets the color of the Light")
        , lightIntensity("Intensity", "Intensity of the Light") {

    this->getLightSlot.SetCompatibleCall<CallLightDescription>();
    this->MakeSlotAvailable(&this->getLightSlot);

    this->deployLightSlot.SetCallback(
        CallLight::ClassName(), CallLight::FunctionName(0), &AbstractLight::getLightCallback);
    this->deployLightSlot.SetCallback(
        CallLight::ClassName(), CallLight::FunctionName(1), &AbstractLight::getMetaDataCallback);
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

    LightCollection light_collection;

    if (lc_out != nullptr) {
        if (!(*lc_out)(0)) {
            return false;
        }

        // signal update if rhs connection changed
        if (!rhs_connected) {
            ++version;
        }
        rhs_connected = true;

        // signal update if rhs connection has an update
        if (lc_out->hasUpdate()) {
            ++version;
        }

        light_collection = lc_out->getData();
    } else {
        if (rhs_connected) {
            ++version;
        }
        rhs_connected = false;
    }

    if (this->InterfaceIsDirty()) {
        ++version;
    }
    if (lc_in != NULL) {
        this->readParams();
        this->addLight(light_collection);
        lc_in->setData(light_collection, version);
    }

    return true;
}

bool megamol::core::view::light::AbstractLight::getMetaDataCallback(core::Call& call) {
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
