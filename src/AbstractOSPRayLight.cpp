/*
* AbstractOSPRayLight.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "AbstractOSPRayLight.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/sys/Log.h"



using namespace megamol::ospray;


AbstractOSPRayLight::AbstractOSPRayLight(void) :
    core::Module(),
    lightContainer(),
    getLightSlot("getLightSlot", "Connects to the OSPRayRenderer or another OSPRayLight"),
    deployLightSlot("deployLightSlot", "Connects to the OSPRayRenderer or another OSPRayLight"),
    // General light parameters
    lightColor("Light::General::LightColor", "Sets the color of the Light"),
    lightIntensity("Light::General::LightIntensity", "Intensity of the Light")
{
    this->getLightSlot.SetCompatibleCall<CallOSPRayLightDescription>();
    this->MakeSlotAvailable(&this->getLightSlot);

    this->deployLightSlot.SetCallback(CallOSPRayLight::ClassName(), CallOSPRayLight::FunctionName(0), &AbstractOSPRayLight::getLightCallback);
    this->MakeSlotAvailable(&this->deployLightSlot);

    // general light
    this->lightColor << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 1.0f, 1.0f));
    this->lightIntensity << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->lightIntensity);
    this->MakeSlotAvailable(&this->lightColor);
}

AbstractOSPRayLight::~AbstractOSPRayLight(void) {
    AbstractOSPRayLight::release();
}

bool AbstractOSPRayLight::create() {
    this->lightContainer.isValid = true;
    return true;
}

void AbstractOSPRayLight::release() {
    lightContainer.isValid = false;
}


bool AbstractOSPRayLight::getLightCallback(megamol::core::Call& call) {
    CallOSPRayLight *lc_in = dynamic_cast<CallOSPRayLight*>(&call);
    CallOSPRayLight *lc_out = this->getLightSlot.CallAs<CallOSPRayLight>();

    if (lc_in != NULL) {
        if (InterfaceIsDirty()) {
            this->readParams();
            lc_in->addLight(lightContainer);
        }
    }

    if (lc_out != NULL) {
        lc_out=lc_in;
        lc_out->fillLightMap();
    }

    return true;
}

bool AbstractOSPRayLight::AbstractIsDirty() {
    if (
        this->lightIntensity.IsDirty() ||
        this->lightColor.IsDirty() 
        ) {
        this->lightIntensity.ResetDirty();
        this->lightColor.ResetDirty();
        return true;
    } else {
        return false;
    }
}
