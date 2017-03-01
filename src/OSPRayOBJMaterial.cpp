/*
* OSPRayOBJMaterial.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayOBJMaterial.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::ospray;


OSPRayOBJMaterial::OSPRayOBJMaterial(void) :
    AbstractOSPRayMaterial(),
    // Distant light parameters
    // OBJMATERIAL
    Kd("DiffuseColor", "Diffuse color"),
    Ks("SpecularColor", "Specular color"),
    Ns("Shininess", "Phong exponent"),
    d("Opacity", "Opacity"),
    Tf("TransparencyFilterColor", "Transparency filter color") {

    this->Kd << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.8f, 0.8f, 0.8f));
    this->Ks << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->Ns << new core::param::FloatParam(10.0f);
    this->d << new core::param::FloatParam(1.0f);
    this->Tf << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->MakeSlotAvailable(&this->Kd);
    this->MakeSlotAvailable(&this->Ks);
    this->MakeSlotAvailable(&this->Ns);
    this->MakeSlotAvailable(&this->d);
    this->MakeSlotAvailable(&this->Tf);
}

OSPRayOBJMaterial::~OSPRayOBJMaterial(void) {
    // empty
}

void OSPRayOBJMaterial::readParams() {
    materialContainer.materialType = materialTypeEnum::OBJMATERIAL;

    auto kd = this->Kd.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    materialContainer.Kd.assign(kd, kd + 3);

    auto ks = this->Ks.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    materialContainer.Ks.assign(ks, ks + 3);

    auto tf = this->Tf.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    materialContainer.Tf.assign(tf, tf + 3);

    materialContainer.Ns = this->Ns.Param<core::param::FloatParam>()->Value();

    materialContainer.d = this->d.Param<core::param::FloatParam>()->Value();
}

bool OSPRayOBJMaterial::InterfaceIsDirty() {
    if (
        this->Kd.IsDirty() ||
        this->Ks.IsDirty() ||
        this->Ns.IsDirty() ||
        this->d.IsDirty() ||
        this->Tf.IsDirty()
        ) {
        this->Kd.ResetDirty();
        this->Ks.ResetDirty();
        this->Ns.ResetDirty();
        this->d.ResetDirty();
        this->Tf.ResetDirty();
        return true;
    } else {
        return false;
    }
}