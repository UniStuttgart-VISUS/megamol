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
    , flexPosSlot("xyz", "")
    , flexColSlot("i", "")
    , flexBoxSlot("box", "")
    , flexXSlot("x","")
    , flexYSlot("y","")
    , flexZSlot("z","") {

    this->mpSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &ADIOSFlexConvert::getDataCallback);
    this->mpSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &ADIOSFlexConvert::getExtentCallback);
    this->MakeSlotAvailable(&this->mpSlot);

    this->adiosSlot.SetCompatibleCall<CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->adiosSlot);

    this->flexPosSlot << new core::param::FlexEnumParam("undef");
    this->flexPosSlot.SetUpdateCallback(&ADIOSFlexConvert::paramChanged);
    this->MakeSlotAvailable(&this->flexPosSlot);

    this->flexColSlot << new core::param::FlexEnumParam("undef");
    this->flexColSlot.SetUpdateCallback(&ADIOSFlexConvert::paramChanged);
    this->MakeSlotAvailable(&this->flexColSlot);

    this->flexBoxSlot << new core::param::FlexEnumParam("undef");
    this->flexBoxSlot.SetUpdateCallback(&ADIOSFlexConvert::paramChanged);
    this->MakeSlotAvailable(&this->flexBoxSlot);

    this->flexXSlot << new core::param::FlexEnumParam("undef");
    this->flexXSlot.SetUpdateCallback(&ADIOSFlexConvert::paramChanged);
    this->MakeSlotAvailable(&this->flexXSlot);

    this->flexYSlot << new core::param::FlexEnumParam("undef");
    this->flexYSlot.SetUpdateCallback(&ADIOSFlexConvert::paramChanged);
    this->MakeSlotAvailable(&this->flexYSlot);

    this->flexZSlot << new core::param::FlexEnumParam("undef");
    this->flexZSlot.SetUpdateCallback(&ADIOSFlexConvert::paramChanged);
    this->MakeSlotAvailable(&this->flexZSlot);
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
            this->flexPosSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexColSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexBoxSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexXSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexYSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexZSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        }

        cad->setFrameIDtoLoad(mpdc->FrameID());

        const std::string pos_str = std::string(this->flexPosSlot.Param<core::param::FlexEnumParam>()->ValueString());
        const std::string col_str = std::string(this->flexColSlot.Param<core::param::FlexEnumParam>()->ValueString());
        const std::string box_str = std::string(this->flexBoxSlot.Param<core::param::FlexEnumParam>()->ValueString());
        const std::string x_str = std::string(this->flexXSlot.Param<core::param::FlexEnumParam>()->ValueString());
        const std::string y_str = std::string(this->flexYSlot.Param<core::param::FlexEnumParam>()->ValueString());
        const std::string z_str = std::string(this->flexZSlot.Param<core::param::FlexEnumParam>()->ValueString());

        if (pos_str != "undef") {
            if (!cad->inquire(pos_str)) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[ADIOSFlexConvert] variable \"%s\" does not exist.", pos_str.c_str());
            }
        }
        if (col_str != "undef") {
            if (!cad->inquire(col_str)) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[ADIOSFlexConvert] variable \"%s\" does not exist.", col_str.c_str());
            }
        }
        if (box_str != "undef") {
            if (!cad->inquire(box_str)) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[ADIOSFlexConvert] variable \"%s\" does not exist.", box_str.c_str());
            }
        }

        if (x_str != "undef" || y_str != "undef" || z_str != "undef") {
            if (!cad->inquire(x_str)) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[ADIOSFlexConvert] variable \"%s\" does not exist.", x_str.c_str());
            }

            if (!cad->inquire(y_str)) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[ADIOSFlexConvert] variable \"%s\" does not exist.", y_str.c_str());
            }

            if (!cad->inquire(z_str)) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[ADIOSFlexConvert] variable \"%s\" does not exist.", z_str.c_str());
            }
        } else {
            if (pos_str == "undef") {
                vislib::sys::Log::DefaultLog.WriteError("[ADIOSFlexConvert] No positions set");
                return false;
            }
        }
  
        if (!(*cad)(0)) {
            vislib::sys::Log::DefaultLog.WriteError("[ADIOSFlexConvert] Error during GetData");
            return false;
        }

        std::vector<float> XYZ;
        std::vector<float> X;
        std::vector<float> Y;
        std::vector<float> Z;
        uint64_t p_count;
        stride = 0;
        if (pos_str != "undef") {
            XYZ = cad->getData(pos_str)->GetAsFloat();
            p_count = XYZ.size() / 3;
            stride += 3 * sizeof(float);
        } else if (x_str != "undef" || y_str != "undef" || z_str != "undef") {
            X = cad->getData(x_str)->GetAsFloat();
            Y = cad->getData(y_str)->GetAsFloat();
            Z = cad->getData(z_str)->GetAsFloat();
            p_count = X.size();
            stride += 3 * sizeof(float);
        } else { return false; }

        std::vector<float> col;
        if (col_str != "undef") {
            col = cad->getData(col_str)->GetAsFloat();
            stride += 1 * sizeof(float);
        }

        const int float_step = stride / sizeof(float);
        // get bounding box
        vislib::math::Cuboid<float> cubo;
        if (box_str != "undef") {
            auto box = cad->getData(box_str)->GetAsFloat();
            cubo = vislib::math::Cuboid<float>(box[0], 
                box[1], std::min(box[5], box[2]),
                box[3], box[4], std::max(box[5], box[2]));
        }

        mpdc->SetParticleListCount(1);
        mix.clear();
        if (col_str != "undef") {
            mix.resize(p_count * 4);
            colType = core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I;
        } else {
            mix.resize(p_count * 3);
            colType = core::moldyn::SimpleSphericalParticles::COLDATA_NONE;
        }

        // Set types
        vertType = core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ;
        idType = core::moldyn::SimpleSphericalParticles::IDDATA_NONE;

        mpdc->AccessParticles(0).SetGlobalRadius(0.1);

        float xmin = std::numeric_limits<float>::max();
        float xmax = std::numeric_limits<float>::min();
        float ymin = std::numeric_limits<float>::max();
        float ymax = std::numeric_limits<float>::min();
        float zmin = std::numeric_limits<float>::max();
        float zmax = std::numeric_limits<float>::min();
        for (size_t i = 0; i < p_count; i++) {
            if (pos_str != "undef") {
                mix[float_step * i + 0] = XYZ[3 * i + 0];
                mix[float_step * i + 1] = XYZ[3 * i + 1];
                mix[float_step * i + 2] = XYZ[3 * i + 2];
            } else {
                mix[float_step * i + 0] = X[i];
                mix[float_step * i + 1] = Y[i];
                mix[float_step * i + 2] = Z[i];
            }
            if (col_str != "undef") {
                mix[float_step * i + 3] = col[i];
            }
            if (box_str == "undef") {
                xmin = std::min(xmin, mix[float_step * i + 0]);
                xmax = std::max(xmax, mix[float_step * i + 0]);
                ymin = std::min(ymin, mix[float_step * i + 1]);
                ymax = std::max(ymax, mix[float_step * i + 1]);
                zmin = std::min(zmin, mix[float_step * i + 2]);
                zmax = std::max(zmax, mix[float_step * i + 2]);
            }
        }
        if (box_str == "undef") {
            cubo = vislib::math::Cuboid<float>(xmin, ymin, zmin, xmax, ymax, zmax);
        }

        // Set bounding box
        mpdc->AccessBoundingBoxes().SetObjectSpaceBBox(cubo);
        mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox(cubo);


        // Set particles
        mpdc->AccessParticles(0).SetCount(p_count);

        mpdc->AccessParticles(0).SetVertexData(vertType, mix.data(), stride);
        //mpdc->AccessParticles(0).SetColourData(
        //    colType, mix.data() + core::moldyn::SimpleSphericalParticles::VertexDataSize[vertType], stride);
        mpdc->AccessParticles(0).SetColourData(
            colType, &mix[3], stride);

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