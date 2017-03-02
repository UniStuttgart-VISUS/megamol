/*
* CallOSPRayMaterial.cpp
*
* Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "CallOSPRayMaterial.h"
#include "vislib/IllegalParamException.h"

using namespace megamol::ospray;

OSPRayMaterialContainer::OSPRayMaterialContainer() :
    materialType(materialTypeEnum::OBJMATERIAL),
    // OBJMaterial/ScivisMaterial
    Kd(NULL),
    Ks(NULL),
    Ns(0.0f),
    d(0.0f),
    Tf(0.0f),
    // LUMINOUS
    lumColor(0.0f),
    lumIntensity(0.0f),
    lumTransparency(0.0f),
    // VELVET
    velvetReflectance(NULL),
    velvetBackScattering(0.0f),
    velvetHorizonScatteringColor(NULL),
    velvetHorizonScatteringFallOff(0.0f),
    // MATTE
    matteReflectance(NULL),
    // METAL
    metalReflectance(NULL),
    metalEta(NULL),
    metalK(NULL),
    metalRoughness(0.0f),
    // METALLICPAINT
    metallicShadeColor(NULL),
    metallicGlitterColor(NULL),
    metallicGlitterSpread(0.0f),
    metallicEta(0.0f),
    // GLASS
    glassEtaInside(0.0f),
    glassEtaOutside(0.0f),
    glassAttenuationColorInside(NULL),
    glassAttenuationColorOutside(NULL),
    glassAttenuationDistance(0.0f),
    //THINGLASS
    thinglassTransmission(NULL),
    thinglassEta(0.0f),
    thinglassThickness(0.0f),
    // PLASTIC
    plasticPigmentColor(NULL),
    plasticEta(0.0f),
    plasticRoughness(0.0f),
    plasticThickness(0.0f),

    isValid(false) {}

OSPRayMaterialContainer::~OSPRayMaterialContainer() {
    //
}

// ################################
// ###### CallOSPRayMaterial ######
// ################################
/*
* megamol::ospray::CallOSPRayLight::CallOSPRayLight
*/
CallOSPRayMaterial::CallOSPRayMaterial(void) : isDirty(false) {
    // intentionally empty
}

/*
* megamol::ospray::CallOSPRayLight::~CallOSPRayLight
*/
CallOSPRayMaterial::~CallOSPRayMaterial(void) {
    //
}

/*
* megamol::ospray::CallOSPRayLight::operator=
*/
CallOSPRayMaterial& CallOSPRayMaterial::operator=(const CallOSPRayMaterial& rhs) {
    return *this;
}

void CallOSPRayMaterial::setMaterialContainer(std::shared_ptr<OSPRayMaterialContainer> mc) {
    this->materialContainer = mc;
}

std::shared_ptr<OSPRayMaterialContainer> CallOSPRayMaterial::getMaterialParameter() {
    if (!(*this)(0)) {
        throw vislib::IllegalParamException("Error in fillMaterialContainer", __FILE__, __LINE__);
    }
    return std::move(this->materialContainer);
}

bool CallOSPRayMaterial::InterfaceIsDirty() {
    if (this->isDirty) {
        this->isDirty = false;
        return true;
    }
    return false;
}

void CallOSPRayMaterial::setDirty() {
    this->isDirty = true;
}