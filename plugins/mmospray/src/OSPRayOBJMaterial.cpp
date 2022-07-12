/*
 * OSPRayOBJMaterial.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "OSPRayOBJMaterial.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"

using namespace megamol::ospray;


OSPRayOBJMaterial::OSPRayOBJMaterial(void)
        : AbstractOSPRayMaterial()
        ,
        // Distant light parameters
        // OBJMATERIAL

        Kd("DiffuseColor", "Diffuse color")
        , Ks("SpecularColor", "Specular color")
        , Ns("Shininess", "Phong exponent")
        , d("Opacity", "Opacity")
        , Tf("TransparencyFilterColor", "Transparency filter color") {

    this->Kd << new core::param::ColorParam(0.8f * 255, 0.8f * 255, 0.8f * 255, 1.0f * 255);
    this->Ks << new core::param::ColorParam(0, 0, 0, 1);
    this->Ns << new core::param::FloatParam(10.0f);
    this->d << new core::param::FloatParam(1.0f);
    this->Tf << new core::param::ColorParam(0, 0, 0, 1);
    this->MakeSlotAvailable(&this->Kd);
    this->MakeSlotAvailable(&this->Ks);
    this->MakeSlotAvailable(&this->Ns);
    this->MakeSlotAvailable(&this->d);
    this->MakeSlotAvailable(&this->Tf);
}

OSPRayOBJMaterial::~OSPRayOBJMaterial(void) {
    this->Release();
}

void OSPRayOBJMaterial::readParams() {
    materialContainer.materialType = materialTypeEnum::OBJMATERIAL;

    objMaterial obj;

    auto kd = this->Kd.Param<core::param::ColorParam>()->GetArray();
    obj.Kd = {kd[0], kd[1], kd[2]};
    auto ks = this->Ks.Param<core::param::ColorParam>()->GetArray();
    obj.Ks = {ks[0], ks[1], ks[2]};
    auto tf = this->Tf.Param<core::param::ColorParam>()->GetArray();
    obj.Tf = {tf[0], tf[1], tf[2]};
    obj.Ns = this->Ns.Param<core::param::FloatParam>()->Value();
    obj.d = this->d.Param<core::param::FloatParam>()->Value();

    materialContainer.material = obj;
}

bool OSPRayOBJMaterial::InterfaceIsDirty() {
    if (this->Kd.IsDirty() || this->Ks.IsDirty() || this->Ns.IsDirty() || this->d.IsDirty() || this->Tf.IsDirty()) {
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
