/*
 * ADIOSFlexConvert.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ADIOSFlexConvert.h"
#include <complex.h>
#include <numeric>
#include "adios_plugin/CallADIOSData.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/FlexEnumParam.h"
#include "vislib/sys/Log.h"

namespace megamol {
namespace adios {

ADIOSFlexConvert::ADIOSFlexConvert()
    : core::Module()
    , mpSlot("mpSlot", "Slot to send multi particle data.")
    , adiosSlot("adiosSlot", "Slot to request ADIOS IO")
    , flexPos("xyz", "")
    , flexCol("i", "")
    , flexBox("box", "") {

    this->mpSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &ADIOSFlexConvert::getDataCallback);
    this->mpSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &ADIOSFlexConvert::getExtentCallback);
    this->MakeSlotAvailable(&this->mpSlot);

    this->adiosSlot.SetCompatibleCall<CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->adiosSlot);

    core::param::FlexEnumParam* posEnum = new core::param::FlexEnumParam("undef");
    this->flexPos << posEnum;
    this->flexPos.SetUpdateCallback(&ADIOSFlexConvert::paramChanged);
    this->MakeSlotAvailable(&this->flexPos);

    core::param::FlexEnumParam* colEnum = new core::param::FlexEnumParam("undef");
    this->flexCol << colEnum;
    this->flexCol.SetUpdateCallback(&ADIOSFlexConvert::paramChanged);
    this->MakeSlotAvailable(&this->flexCol);

    core::param::FlexEnumParam* boxEnum = new core::param::FlexEnumParam("undef");
    this->flexBox << boxEnum;
    this->flexBox.SetUpdateCallback(&ADIOSFlexConvert::paramChanged);
    this->MakeSlotAvailable(&this->flexBox);
}

ADIOSFlexConvert::~ADIOSFlexConvert() { this->Release(); }

bool ADIOSFlexConvert::create() { return true; }

void ADIOSFlexConvert::release() {}

bool ADIOSFlexConvert::getDataCallback(core::Call& call) {
    core::moldyn::MultiParticleDataCall* mpdc = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
    if (mpdc == nullptr) return false;

    CallADIOSData* cad = this->adiosSlot.CallAs<CallADIOSData>();
    if (cad == nullptr) return false;

    if (!(*cad)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("[ADIOSFlexConvert] Error during GetHeader");
        return false;
    }
    bool dathashChanged = (mpdc->DataHash() != cad->getDataHash());
    if ((mpdc->FrameID() != currentFrame) || dathashChanged || _trigger_recalc) {

        _trigger_recalc = false;

        // get adios meta data
        auto availVars = cad->getAvailableVars();
        for (auto var : availVars) {
            this->flexPos.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexCol.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexBox.Param<core::param::FlexEnumParam>()->AddValue(var);
        }

        cad->setFrameIDtoLoad(mpdc->FrameID());

        const std::string pos_str = std::string(this->flexPos.Param<core::param::FlexEnumParam>()->ValueString());
        const std::string col_str = std::string(this->flexCol.Param<core::param::FlexEnumParam>()->ValueString());
        const std::string box_str = std::string(this->flexBox.Param<core::param::FlexEnumParam>()->ValueString());

        if (!cad->inquire(pos_str)) {
            vislib::sys::Log::DefaultLog.WriteError(
                "[ADIOSFlexConvert] variable \"%s\" does not exist.", pos_str.c_str());
        }

        if (!cad->inquire(col_str)) {
            vislib::sys::Log::DefaultLog.WriteError(
                "[ADIOSFlexConvert] variable \"%s\" does not exist.", col_str.c_str());
        }

        if (!cad->inquire(box_str)) {
            vislib::sys::Log::DefaultLog.WriteError(
                "[ADIOSFlexConvert] variable \"%s\" does not exist.", box_str.c_str());
        }

        cad->inquire("global_radius");
        cad->inquire("p_count");

        if (!(*cad)(0)) {
            vislib::sys::Log::DefaultLog.WriteError("[ADIOSFlexConvert] Error during GetData");
            return false;
        }


        stride = 0;
        auto XYZ = cad->getData(pos_str)->GetAsDouble();
        stride += 3 * sizeof(float);

        auto radius = cad->getData("global_radius")->GetAsDouble();
        auto box = cad->getData(box_str)->GetAsDouble();
        auto p_count = cad->getData("p_count")->GetAsUInt32();

        auto col = cad->getData(col_str)->GetAsDouble();
        stride += 1 * sizeof(float);

        // Set bounding box
        const vislib::math::Cuboid<float> cubo(box[0], box[1], std::min(box[5],box[2]), box[3], box[4], std::max(box[5],box[2]));
        mpdc->AccessBoundingBoxes().SetObjectSpaceBBox(cubo);
        mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox(cubo);


        mpdc->SetParticleListCount(1);
        mix.clear();
        mix.reserve(p_count[0] * 4);

        // Set types
        colType = core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I;
        vertType = core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ;
        idType = core::moldyn::SimpleSphericalParticles::IDDATA_NONE;

        mpdc->AccessParticles(0).SetGlobalRadius(radius[0]);

        for (size_t i = 0; i < p_count[0]; i++) {
            mix.emplace_back(XYZ[3 * i + 0]);
            mix.emplace_back(XYZ[3 * i + 1]);
            mix.emplace_back(XYZ[3 * i + 2]);
            mix.emplace_back(col[i]);
        }


        // Set particles
        mpdc->AccessParticles(0).SetCount(p_count[0]);

        mpdc->AccessParticles(0).SetVertexData(vertType, mix.data(), stride);
        mpdc->AccessParticles(0).SetColourData(
            colType, mix.data() + core::moldyn::SimpleSphericalParticles::VertexDataSize[vertType], stride);
        mpdc->AccessParticles(0).SetIDData(idType,
            mix.data() + core::moldyn::SimpleSphericalParticles::VertexDataSize[vertType] +
                core::moldyn::SimpleSphericalParticles::ColorDataSize[colType],
            stride);
        mpdc->AccessParticles(0).SetBBox(cubo);
    }

    mpdc->SetFrameCount(cad->getFrameCount());
    mpdc->SetDataHash(cad->getDataHash());
    currentFrame = mpdc->FrameID();

    return true;
}

bool ADIOSFlexConvert::getExtentCallback(core::Call& call) {

    core::moldyn::MultiParticleDataCall* mpdc = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
    if (mpdc == nullptr) return false;

    CallADIOSData* cad = this->adiosSlot.CallAs<CallADIOSData>();
    if (cad == nullptr) return false;

    if (!this->getDataCallback(call)) return false;

    return true;
}

bool ADIOSFlexConvert::paramChanged(core::param::ParamSlot& p) {

    _trigger_recalc = true;
    return true;
}

} // end namespace adios
} // end namespace megamol