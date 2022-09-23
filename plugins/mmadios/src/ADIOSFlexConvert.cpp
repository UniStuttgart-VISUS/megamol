/*
 * ADIOSFlexConvert.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "ADIOSFlexConvert.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmadios/CallADIOSData.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/utility/log/Log.h"
#include <numeric>

namespace megamol {
namespace adios {

ADIOSFlexConvert::ADIOSFlexConvert()
        : core::Module()
        , mpSlot("mpSlot", "Slot to send multi particle data.")
        , adiosSlot("adiosSlot", "Slot to request ADIOS IO")
        , flexPosSlot("xyz", "")
        , flexColSlot("i", "")
        , flexBoxSlot("box", "")
        , flexXSlot("x", "")
        , flexYSlot("y", "")
        , flexZSlot("z", "")
        , flexAlignedPosSlot("xyzw", "")
        , flexIDSlot("id", "The name of the data holding the particle id.")
        , flexVXSlot("direction::vx", "The name of the data holding the vx-coordinate.")
        , flexVYSlot("direction::vy", "The name of the data holding the vy-coordinate.")
        , flexVZSlot("direction::vz", "The name of the data holding the vz-coordinate.") {

    this->mpSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(0), &ADIOSFlexConvert::getDataCallback);
    this->mpSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(1), &ADIOSFlexConvert::getExtentCallback);
    this->MakeSlotAvailable(&this->mpSlot);

    this->adiosSlot.SetCompatibleCall<CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->adiosSlot);

    this->flexAlignedPosSlot << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->flexAlignedPosSlot);

    this->flexPosSlot << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->flexPosSlot);

    this->flexColSlot << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->flexColSlot);

    this->flexBoxSlot << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->flexBoxSlot);

    this->flexXSlot << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->flexXSlot);

    this->flexYSlot << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->flexYSlot);

    this->flexZSlot << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->flexZSlot);

    this->flexIDSlot << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->flexIDSlot);

    this->flexVXSlot << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->flexVXSlot);

    this->flexVYSlot << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->flexVYSlot);

    this->flexVZSlot << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->flexVZSlot);
}

ADIOSFlexConvert::~ADIOSFlexConvert() {
    this->Release();
}

bool ADIOSFlexConvert::create() {
    return true;
}

void ADIOSFlexConvert::release() {}


bool ADIOSFlexConvert::inquireDataVariables(CallADIOSData* cad) {
    const auto pos_str = this->flexPosSlot.Param<core::param::FlexEnumParam>()->ValueString();
    const auto apos_str = this->flexAlignedPosSlot.Param<core::param::FlexEnumParam>()->ValueString();
    const auto col_str = this->flexColSlot.Param<core::param::FlexEnumParam>()->ValueString();
    const auto box_str = this->flexBoxSlot.Param<core::param::FlexEnumParam>()->ValueString();
    const auto x_str = this->flexXSlot.Param<core::param::FlexEnumParam>()->ValueString();
    const auto y_str = this->flexYSlot.Param<core::param::FlexEnumParam>()->ValueString();
    const auto z_str = this->flexZSlot.Param<core::param::FlexEnumParam>()->ValueString();
    const auto id_str = this->flexIDSlot.Param<core::param::FlexEnumParam>()->ValueString();
    const auto vx_str = this->flexVXSlot.Param<core::param::FlexEnumParam>()->ValueString();
    const auto vy_str = this->flexVYSlot.Param<core::param::FlexEnumParam>()->ValueString();
    const auto vz_str = this->flexVZSlot.Param<core::param::FlexEnumParam>()->ValueString();

    if (pos_str != "undef") {
        if (!cad->inquireVar(pos_str)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ADIOSFlexConvert] variable \"%s\" does not exist.", pos_str.c_str());
        }
    }
    if (col_str != "undef") {
        if (!cad->inquireVar(col_str)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ADIOSFlexConvert] variable \"%s\" does not exist.", col_str.c_str());
        }
    }

    if (x_str != "undef" || y_str != "undef" || z_str != "undef") {
        if (!cad->inquireVar(x_str)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ADIOSFlexConvert] variable \"%s\" does not exist.", x_str.c_str());
        }

        if (!cad->inquireVar(y_str)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ADIOSFlexConvert] variable \"%s\" does not exist.", y_str.c_str());
        }

        if (!cad->inquireVar(z_str)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ADIOSFlexConvert] variable \"%s\" does not exist.", z_str.c_str());
        }
    } else if (pos_str != "undef") {
        if (!cad->inquireVar(pos_str)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ADIOSFlexConvert] variable \"%s\" does not exist.", pos_str.c_str());
        }
    } else if (apos_str != "undef") {
        if (!cad->inquireVar(apos_str)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ADIOSFlexConvert] variable \"%s\" does not exist.", apos_str.c_str());
        }
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[ADIOSFlexConvert] No positions set");
        return false;
    }

    if (id_str != "undef") {
        if (!cad->inquireVar(id_str)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ADIOSFlexConvert] variable \"%s\" does not exist.", id_str.c_str());
        } else {
            hasID = true;
        }
    }

    if (vx_str != "undef" || vy_str != "undef" || vz_str != "undef") {
        hasVel = true;
        if (!cad->inquireVar(vx_str)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ADIOSFlexConvert] variable \"%s\" does not exist.", vx_str.c_str());
            hasVel = false;
        }

        if (!cad->inquireVar(vy_str)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ADIOSFlexConvert] variable \"%s\" does not exist.", vy_str.c_str());
            hasVel = false;
        }

        if (!cad->inquireVar(vz_str)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ADIOSFlexConvert] variable \"%s\" does not exist.", vz_str.c_str());
            hasVel = false;
        }
    }
    return true;
}

bool ADIOSFlexConvert::inquireMetaDataVariables(CallADIOSData* cad) {
    if (this->flexBoxSlot.Param<core::param::FlexEnumParam>()->ValueString() != "undef") {
        if (!cad->inquireVar(this->flexBoxSlot.Param<core::param::FlexEnumParam>()->ValueString())) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ADIOSFlexConvert] variable \"%s\" does not exist.",
                this->flexBoxSlot.Param<core::param::FlexEnumParam>()->ValueString().c_str());
        }
        return false;
    }
    return true;
}


bool ADIOSFlexConvert::getDataCallback(core::Call& call) {
    geocalls::MultiParticleDataCall* mpdc = dynamic_cast<geocalls::MultiParticleDataCall*>(&call);
    if (mpdc == nullptr)
        return false;

    CallADIOSData* cad = this->adiosSlot.CallAs<CallADIOSData>();
    if (cad == nullptr)
        return false;

    if (!(*cad)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[ADIOSFlexConvert] Error during GetHeader");
        return false;
    }
    auto anythingDirty = this->AnyParameterDirty();
    bool dathashChanged = (mpdc->DataHash() != cad->getDataHash());
    if ((mpdc->FrameID() != currentFrame) || dathashChanged || anythingDirty) {

        this->ResetAllDirtyFlags();

        // get adios meta data
        auto availVars = cad->getAvailableVars();
        for (auto var : availVars) {
            this->flexPosSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexAlignedPosSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexColSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexBoxSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexXSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexYSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexZSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexIDSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexVXSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexVYSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexVZSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        }

        cad->setFrameIDtoLoad(mpdc->FrameID());

        hasVel = hasID = false;

        this->inquireMetaDataVariables(cad);
        this->inquireDataVariables(cad);

        const auto pos_str = this->flexPosSlot.Param<core::param::FlexEnumParam>()->ValueString();
        const auto apos_str = this->flexAlignedPosSlot.Param<core::param::FlexEnumParam>()->ValueString();
        const auto col_str = this->flexColSlot.Param<core::param::FlexEnumParam>()->ValueString();
        const auto box_str = this->flexBoxSlot.Param<core::param::FlexEnumParam>()->ValueString();
        const auto x_str = this->flexXSlot.Param<core::param::FlexEnumParam>()->ValueString();
        const auto y_str = this->flexYSlot.Param<core::param::FlexEnumParam>()->ValueString();
        const auto z_str = this->flexZSlot.Param<core::param::FlexEnumParam>()->ValueString();
        const auto id_str = this->flexIDSlot.Param<core::param::FlexEnumParam>()->ValueString();
        const auto vx_str = this->flexVXSlot.Param<core::param::FlexEnumParam>()->ValueString();
        const auto vy_str = this->flexVYSlot.Param<core::param::FlexEnumParam>()->ValueString();
        const auto vz_str = this->flexVZSlot.Param<core::param::FlexEnumParam>()->ValueString();

        if (!(*cad)(0)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("[ADIOSFlexConvert] Error during GetData");
            return false;
        }

        std::vector<float> XYZ;
        std::vector<float> XYZW;
        std::vector<float> X;
        std::vector<float> Y;
        std::vector<float> Z;
        std::vector<float> VX;
        std::vector<float> VY;
        std::vector<float> VZ;
        std::vector<uint32_t> ID;
        uint64_t p_count;
        stride = 0;
        if (pos_str != "undef") {
            XYZ = cad->getData(pos_str)->GetAsFloat();
            p_count = XYZ.size() / 3;
            stride += 3 * sizeof(float);
        } else if (x_str != "undef" && y_str != "undef" && z_str != "undef") {
            X = cad->getData(x_str)->GetAsFloat();
            Y = cad->getData(y_str)->GetAsFloat();
            Z = cad->getData(z_str)->GetAsFloat();
            p_count = X.size();
            stride += 3 * sizeof(float);
        } else if (apos_str != "undef") {
            XYZW = cad->getData(apos_str)->GetAsFloat();
            p_count = XYZW.size() / 4;
            stride += 3 * sizeof(float);
        } else {
            return false;
        }

        if (hasVel) {
            VX = cad->getData(vx_str)->GetAsFloat();
            VY = cad->getData(vy_str)->GetAsFloat();
            VZ = cad->getData(vz_str)->GetAsFloat();
            stride += 3 * sizeof(float);
        }

        if (hasID) {
            ID = cad->getData(id_str)->GetAsUInt32();
            stride += sizeof(uint32_t);
        }

        std::vector<float> col;
        if (col_str != "undef") {
            col = cad->getData(col_str)->GetAsFloat();
            stride += 1 * sizeof(float);
        }

        const int float_step = stride / sizeof(float);

        mpdc->SetParticleListCount(1);
        mix.clear();
        mix.resize(p_count * stride);
        if (col_str != "undef") {
            colType = geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I;
        } else {
            colType = geocalls::SimpleSphericalParticles::COLDATA_NONE;
        }

        // Set types
        vertType = geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ;
        if (hasID) {
            idType = geocalls::SimpleSphericalParticles::IDDATA_UINT32;
        } else {
            idType = geocalls::SimpleSphericalParticles::IDDATA_NONE;
        }
        if (hasVel) {
            dirType = geocalls::SimpleSphericalParticles::DIRDATA_FLOAT_XYZ;
        } else {
            dirType = geocalls::SimpleSphericalParticles::DIRDATA_NONE;
        }


        mpdc->AccessParticles(0).SetGlobalRadius(0.1);

        float xmin = std::numeric_limits<float>::max();
        float xmax = std::numeric_limits<float>::lowest();
        float ymin = std::numeric_limits<float>::max();
        float ymax = std::numeric_limits<float>::lowest();
        float zmin = std::numeric_limits<float>::max();
        float zmax = std::numeric_limits<float>::lowest();

        float imin = std::numeric_limits<float>::max();
        float imax = std::numeric_limits<float>::lowest();

        size_t vel_offset = 0;
        size_t id_offset = 0;
        size_t col_offset = 0;

        for (size_t i = 0; i < p_count; i++) {
            if (pos_str != "undef") {
                mix[float_step * i + 0] = XYZ[3 * i + 0];
                mix[float_step * i + 1] = XYZ[3 * i + 1];
                mix[float_step * i + 2] = XYZ[3 * i + 2];
            } else if (x_str != "undef" || y_str != "undef" || z_str != "undef") {
                mix[float_step * i + 0] = X[i];
                mix[float_step * i + 1] = Y[i];
                mix[float_step * i + 2] = Z[i];
            } else if (apos_str != "undef") {
                mix[float_step * i + 0] = XYZW[4 * i + 0];
                mix[float_step * i + 1] = XYZW[4 * i + 1];
                mix[float_step * i + 2] = XYZW[4 * i + 2];
            }

            size_t offset = 3;
            if (col_str != "undef") {
                col_offset = offset;
                mix[float_step * i + 3] = col[i];
                imin = std::min(imin, col[i]);
                imax = std::max(imax, col[i]);
                offset += 1;
            }
            if (box_str == "undef") {
                xmin = std::min(xmin, mix[float_step * i + 0]);
                xmax = std::max(xmax, mix[float_step * i + 0]);
                ymin = std::min(ymin, mix[float_step * i + 1]);
                ymax = std::max(ymax, mix[float_step * i + 1]);
                zmin = std::min(zmin, mix[float_step * i + 2]);
                zmax = std::max(zmax, mix[float_step * i + 2]);
            }

            if (hasVel) {
                vel_offset = offset;
                mix[float_step * i + offset + 0] = VX[i];
                mix[float_step * i + offset + 1] = VY[i];
                mix[float_step * i + offset + 2] = VZ[i];
                offset += 3;
            }

            if (hasID) {
                id_offset = offset;
                std::memcpy(&mix[float_step * i + offset], &ID[i], sizeof(uint32_t));
                offset += 1;
            }
        }
        auto cubo = vislib::math::Cuboid<float>(xmin, ymin, zmin, xmax, ymax, zmax);
        if (box_str == "undef") {
            // Set bounding box
            mpdc->AccessBoundingBoxes().SetObjectSpaceBBox(cubo);
            mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox(cubo);
        }

        // Set particles
        mpdc->AccessParticles(0).SetCount(p_count);

        mpdc->AccessParticles(0).SetVertexData(vertType, mix.data(), stride);
        //mpdc->AccessParticles(0).SetColourData(
        //    colType, mix.data() + geocalls::SimpleSphericalParticles::VertexDataSize[vertType], stride);
        mpdc->AccessParticles(0).SetColourData(colType, &mix[col_offset], stride);
        mpdc->AccessParticles(0).SetColourMapIndexValues(imin, imax);
        mpdc->AccessParticles(0).SetIDData(idType, &mix[id_offset], stride);
        mpdc->AccessParticles(0).SetDirData(dirType, &mix[vel_offset], stride);
        mpdc->AccessParticles(0).SetBBox(cubo);
    }

    mpdc->SetDataHash(cad->getDataHash());
    currentFrame = mpdc->FrameID();

    return true;
}

bool ADIOSFlexConvert::getExtentCallback(core::Call& call) {

    geocalls::MultiParticleDataCall* mpdc = dynamic_cast<geocalls::MultiParticleDataCall*>(&call);
    if (mpdc == nullptr)
        return false;

    CallADIOSData* cad = this->adiosSlot.CallAs<CallADIOSData>();
    if (cad == nullptr)
        return false;

    //if (!this->getDataCallback(call))
    //    return false;

    if (!(*cad)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[ADIOSFlexConvert] Error fetching extents");
        return false;
    }

    cad->setFrameIDtoLoad(mpdc->FrameID());
    this->inquireMetaDataVariables(cad);

    if (!(*cad)(0)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[ADIOSFlexConvert] Error fetching partial data");
        return false;
    }

    // get bounding box
    if (this->flexBoxSlot.Param<core::param::FlexEnumParam>()->ValueString() != "undef") {
        auto the_box = cad->getData(this->flexBoxSlot.Param<core::param::FlexEnumParam>()->ValueString())->GetAsFloat();
        bbox.Set(the_box[0], the_box[1], the_box[2], the_box[3], the_box[4], the_box[5]);
        mpdc->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
        mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox(bbox);
    }

    mpdc->SetFrameCount(std::max<size_t>(cad->getFrameCount(), 1));

    return true;
}

} // end namespace adios
} // end namespace megamol
