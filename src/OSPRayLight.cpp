/*
* OSPRayLight.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRayLight.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/sys/Log.h"



using namespace megamol::ospray;


OSPRayLight::OSPRayLight(void) :
    core::Module(),
    lightContainer(),
    getLightSlot("getLightSlot", "Connects to the OSPRayRenderer or another OSPRayLight"),
    deployLightSlot("deployLightSlot", "Connects to the OSPRayRenderer or another OSPRayLight"),
    // General light parameters
    lightColor("Light::General::LightColor", "Sets the color of the Light"),
    lightIntensity("Light::General::LightIntensity", "Intensity of the Light"),
    lightType("Light::Type::LightType", "Type of the light"),
    // Distant light parameters
    dl_direction("Light::DistantLight::LightDirection", "Direction of the Light"),
    dl_angularDiameter("Light::DistantLight::AngularDiameter", "If greater than zero results in soft shadows"),
    dl_eye_direction("Light::DistantLight::EyeDirection", "Sets the light direction as view direction"),
    // point light parameters
    pl_position("Light::PointLight::Position", ""),
    pl_radius("Light::PointLight::Radius", ""),
    // spot light parameters
    sl_position("Light::SpotLight::Position", ""),
    sl_direction("Light::SpotLight::Direction", ""),
    sl_openingAngle("Light::SpotLight::openingAngle", ""),
    sl_penumbraAngle("Light::SpotLight::penumbraAngle", ""),
    sl_radius("Light::SpotLight::Radius", ""),
    // quad light parameters
    ql_position("Light::QuadLight::Position", ""),
    ql_edgeOne("Light::QuadLight::Edge1", ""),
    ql_edgeTwo("Light::QuadLight::Edge2", ""),
    // hdri light parameteres
    hdri_up("Light::HDRILight::up", ""),
    hdri_direction("Light::HDRILight::Direction", ""),
    hdri_evnfile("Light::HDRILight::EvironmentFile", "")

{
    this->getLightSlot.SetCompatibleCall<CallOSPRayLightDescription>();
    this->MakeSlotAvailable(&this->getLightSlot);

    this->deployLightSlot.SetCallback(CallOSPRayLight::ClassName(), CallOSPRayLight::FunctionName(0), &OSPRayLight::getLightCallback);
    this->MakeSlotAvailable(&this->deployLightSlot);

    core::param::EnumParam *lt = new core::param::EnumParam(lightenum::AMBIENTLIGHT);
    lt->SetTypePair(lightenum::NONE, "None");
    lt->SetTypePair(lightenum::DISTANTLIGHT, "DistantLight");
    lt->SetTypePair(lightenum::POINTLIGHT, "PointLight");
    lt->SetTypePair(lightenum::SPOTLIGHT, "SpotLight");
    lt->SetTypePair(lightenum::QUADLIGHT, "QuadLight");
    lt->SetTypePair(lightenum::AMBIENTLIGHT, "AmbientLight");
    lt->SetTypePair(lightenum::HDRILIGHT, "HDRILight");

    // general light

    this->lightColor << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 1.0f, 1.0f));
    this->lightType << lt;
    this->lightIntensity << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->lightIntensity);
    this->MakeSlotAvailable(&this->lightColor);
    this->MakeSlotAvailable(&this->lightType);

    // distant light
    this->dl_angularDiameter << new core::param::FloatParam(0.0f);
    this->dl_direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, -1.0f, 0.0f));
    this->dl_eye_direction << new core::param::BoolParam(0);
    this->MakeSlotAvailable(&this->dl_direction);
    this->MakeSlotAvailable(&this->dl_angularDiameter);
    this->MakeSlotAvailable(&this->dl_eye_direction);

    // point light
    this->pl_position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->pl_radius << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->pl_position);
    this->MakeSlotAvailable(&this->pl_radius);

    // spot light
    this->sl_position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    this->sl_direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    this->sl_openingAngle << new core::param::FloatParam(0.0f);
    this->sl_penumbraAngle << new core::param::FloatParam(0.0f);
    this->sl_radius << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->sl_position);
    this->MakeSlotAvailable(&this->sl_direction);
    this->MakeSlotAvailable(&this->sl_openingAngle);
    this->MakeSlotAvailable(&this->sl_penumbraAngle);
    this->MakeSlotAvailable(&this->sl_radius);

    // quad light
    this->ql_position << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f));
    this->ql_edgeOne << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    this->ql_edgeTwo << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->ql_position);
    this->MakeSlotAvailable(&this->ql_edgeOne);
    this->MakeSlotAvailable(&this->ql_edgeTwo);

    // HDRI light
    this->hdri_up << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    this->hdri_direction << new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f));
    this->hdri_evnfile << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->hdri_up);
    this->MakeSlotAvailable(&this->hdri_direction);
    this->MakeSlotAvailable(&this->hdri_evnfile);
}

OSPRayLight::~OSPRayLight(void) {
    OSPRayLight::release();
}

bool OSPRayLight::create() {
    this->lightType.Param<core::param::EnumParam>()->setDirty();
    this->lightContainer.isValid = true;
    return true;
}

void OSPRayLight::release() {
    lightContainer.isValid = false;
}


bool OSPRayLight::getLightCallback(megamol::core::Call& call) {
    CallOSPRayLight *lc_in = dynamic_cast<CallOSPRayLight*>(&call);
    CallOSPRayLight *lc_out = this->getLightSlot.CallAs<CallOSPRayLight>();

    if (lc_in != NULL) {
        if (this->InterfaceIsDirty()) {
            this->readParams();
            lc_in->addLight(lightContainer);
            *(lc_in->ModuleIsDirty) = true;
        }
    }

    if (lc_out != NULL) {
        lc_out->setLightMap(lc_in->lightMap);
        lc_out->setDirtyObj(lc_in->ModuleIsDirty);
        lc_out->fillLightMap();
    }

    return true;
}

void OSPRayLight::readParams() {
    lightContainer.lightType = (lightenum)this->lightType.Param<core::param::EnumParam>()->Value();
    auto lcolor = this->lightColor.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.lightColor.assign(lcolor, lcolor + 3);
    lightContainer.lightIntensity = this->lightIntensity.Param<core::param::FloatParam>()->Value();
    lightContainer.dl_eye_direction = this->dl_eye_direction.Param<core::param::BoolParam>()->Value();
    auto dl_dir = this->dl_direction.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.dl_direction.assign(dl_dir, dl_dir + 3);
    lightContainer.dl_angularDiameter = this->dl_angularDiameter.Param<core::param::FloatParam>()->Value();
    auto pl_pos = this->pl_position.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.pl_position.assign(pl_pos, pl_pos + 3);
    lightContainer.pl_radius = this->pl_radius.Param<core::param::FloatParam>()->Value();
    auto sl_pos = this->sl_position.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.sl_position.assign(sl_pos, sl_pos + 3);
    auto sl_dir = this->sl_direction.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.sl_direction.assign(sl_dir, sl_dir + 3);
    lightContainer.sl_openingAngle = this->sl_openingAngle.Param<core::param::FloatParam>()->Value();
    lightContainer.sl_penumbraAngle = this->sl_penumbraAngle.Param<core::param::FloatParam>()->Value();
    lightContainer.sl_radius = this->sl_radius.Param<core::param::FloatParam>()->Value();
    auto ql_pos = this->ql_position.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.ql_position.assign(ql_pos, ql_pos + 3);
    auto ql_e1 = this->ql_edgeOne.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.ql_edgeOne.assign(ql_e1, ql_e1 + 3);
    auto ql_e2 = this->ql_edgeTwo.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.ql_edgeTwo.assign(ql_e2, ql_e2 + 3);
    auto hdriup = this->hdri_up.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.hdri_up.assign(hdriup, hdriup + 3);
    auto hdri_dir = this->hdri_direction.Param<core::param::Vector3fParam>()->Value().PeekComponents();
    lightContainer.hdri_direction.assign(hdri_dir, hdri_dir + 3);
    lightContainer.hdri_evnfile = this->hdri_evnfile.Param<core::param::FilePathParam>()->Value();
}

bool OSPRayLight::InterfaceIsDirty() {
    if (
        this->lightIntensity.IsDirty() ||
        this->lightType.IsDirty() ||
        this->lightColor.IsDirty() ||

        this->dl_angularDiameter.IsDirty() ||
        this->dl_direction.IsDirty() ||
        this->dl_eye_direction.IsDirty() ||

        this->pl_position.IsDirty() ||
        this->pl_radius.IsDirty() ||

        this->sl_position.IsDirty() ||
        this->sl_direction.IsDirty() ||
        this->sl_openingAngle.IsDirty() ||
        this->sl_penumbraAngle.IsDirty() ||
        this->sl_radius.IsDirty() ||

        this->ql_position.IsDirty() ||
        this->ql_edgeOne.IsDirty() ||
        this->ql_edgeTwo.IsDirty() ||

        this->hdri_up.IsDirty() ||
        this->hdri_direction.IsDirty() ||
        this->hdri_evnfile.IsDirty()
        ) {
        this->InterfaceResetDirty();
        return true;
    } else {
        return false;
    }
}

void OSPRayLight::InterfaceResetDirty() {
    this->lightIntensity.ResetDirty();
    this->lightType.ResetDirty();
    this->lightColor.ResetDirty();

    this->dl_angularDiameter.ResetDirty();
    this->dl_direction.ResetDirty();
    this->dl_eye_direction.ResetDirty();

    this->pl_position.ResetDirty();
    this->pl_radius.ResetDirty();

    this->sl_position.ResetDirty();
    this->sl_direction.ResetDirty();
    this->sl_openingAngle.ResetDirty();
    this->sl_penumbraAngle.ResetDirty();
    this->sl_radius.ResetDirty();

    this->ql_position.ResetDirty();
    this->ql_edgeOne.ResetDirty();
    this->ql_edgeTwo.ResetDirty();

    this->hdri_up.ResetDirty();
    this->hdri_direction.ResetDirty();
    this->hdri_evnfile.ResetDirty();
}
