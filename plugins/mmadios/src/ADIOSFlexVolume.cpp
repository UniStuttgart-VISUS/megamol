/*
 * ADIOSFlexVolume.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "ADIOSFlexVolume.h"
#include "mmadios/CallADIOSData.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/utility/log/Log.h"
#include <cmath>
#include <numeric>

using namespace megamol::geocalls;

namespace megamol {
namespace adios {

ADIOSFlexVolume::ADIOSFlexVolume()
        : core::Module()
        , volumeSlot("volumeSlot", "Slot to send volume data.")
        , adiosSlot("adiosSlot", "Slot to request ADIOS IO")
        , flexVelocitySlot("vxvyvz", "Adios variable containing the velocity vector")
        , flexBoxSlot("box", "Adios variable containing the global bounding box")
        , memoryLayoutSlot("memLayout", "dimension increments (fastest to slowest)") {

    this->volumeSlot.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA), &ADIOSFlexVolume::onGetData);
    this->volumeSlot.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_EXTENTS), &ADIOSFlexVolume::onGetExtents);
    this->volumeSlot.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_METADATA), &ADIOSFlexVolume::onGetExtents);
    this->volumeSlot.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_START_ASYNC), &ADIOSFlexVolume::onStartAsync);
    this->volumeSlot.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_STOP_ASYNC), &ADIOSFlexVolume::onStopAsync);
    this->volumeSlot.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_TRY_GET_DATA), &ADIOSFlexVolume::onTryGetData);
    this->MakeSlotAvailable(&this->volumeSlot);

    this->adiosSlot.SetCompatibleCall<CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->adiosSlot);

    this->flexVelocitySlot << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->flexVelocitySlot);

    this->flexBoxSlot << new core::param::FlexEnumParam("undef");
    this->MakeSlotAvailable(&this->flexBoxSlot);

    auto memLayout = new core::param::EnumParam(0);
    memLayout->SetTypePair(0, "xyz");
    memLayout->SetTypePair(1, "zyx");
    this->memoryLayoutSlot.SetParameter(memLayout);
    this->MakeSlotAvailable(&this->memoryLayoutSlot);
}

ADIOSFlexVolume::~ADIOSFlexVolume() {
    this->Release();
}

bool ADIOSFlexVolume::create() {
    return true;
}

void ADIOSFlexVolume::release() {}

bool ADIOSFlexVolume::inquireDataVariables(CallADIOSData* cad) {
    if (this->flexVelocitySlot.Param<core::param::FlexEnumParam>()->ValueString() != "undef") {
        if (!cad->inquireVar(this->flexVelocitySlot.Param<core::param::FlexEnumParam>()->ValueString())) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("[ADIOSFlexVolume] variable \"%s\" does not exist.",
                this->flexVelocitySlot.Param<core::param::FlexEnumParam>()->ValueString().c_str());
        }
    } else {
        return false;
    }
    return true;
}

bool ADIOSFlexVolume::inquireMetaDataVariables(CallADIOSData* cad) {
    if (this->flexBoxSlot.Param<core::param::FlexEnumParam>()->ValueString() != "undef") {
        if (!cad->inquireVar(this->flexBoxSlot.Param<core::param::FlexEnumParam>()->ValueString())) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("[ADIOSFlexVolume] variable \"%s\" does not exist.",
                this->flexBoxSlot.Param<core::param::FlexEnumParam>()->ValueString().c_str());
        }
    }
    return true;
}

bool ADIOSFlexVolume::assertData(geocalls::VolumetricDataCall* vdc, CallADIOSData* cad) {
    bool dathashChanged = (vdc->DataHash() != cad->getDataHash());
    if ((vdc->FrameID() != currentFrame) || dathashChanged || this->AnyParameterDirty()) {
        this->ResetAllDirtyFlags();

        // get adios meta data
        auto availVars = cad->getAvailableVars();
        for (auto var : availVars) {
            this->flexVelocitySlot.Param<core::param::FlexEnumParam>()->AddValue(var);
            this->flexBoxSlot.Param<core::param::FlexEnumParam>()->AddValue(var);
        }
        cad->setFrameIDtoLoad(vdc->FrameID());

        inquireMetaDataVariables(cad);
        if (!inquireDataVariables(cad))
            return false;

        if (!(*cad)(0)) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("[ADIOSFlexVolume] Error during GetData");
            return false;
        }

        if (this->flexVelocitySlot.Param<core::param::FlexEnumParam>()->ValueString() == "undef") {
            return false;
        }

        const auto the_velocities =
            cad->getData(this->flexVelocitySlot.Param<core::param::FlexEnumParam>()->ValueString());
        const std::vector<float> VXVYVZ = the_velocities->GetAsFloat();
        const std::size_t cell_count = VXVYVZ.size() / 3;
        VMAG.resize(cell_count);
        mins.resize(1);
        maxes.resize(1);
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::lowest();
        int xfactor = 1;
        int yfactor = 1;
        int zfactor = 1;
        switch (memoryLayoutSlot.Param<megamol::core::param::EnumParam>()->Value()) {
        case 0: // xyz
            xfactor = 1;
            yfactor = the_velocities->shape[0];
            zfactor = the_velocities->shape[0] * the_velocities->shape[1];
            break;
        case 1: // zyx
            xfactor = the_velocities->shape[1] * the_velocities->shape[2];
            yfactor = the_velocities->shape[1];
            zfactor = 1;
            break;
        }
        for (auto x = 0; x < the_velocities->shape[0]; ++x) {
            for (auto y = 0; y < the_velocities->shape[1]; ++y) {
                for (auto z = 0; z < the_velocities->shape[2]; ++z) {
                    auto offset_in = z * zfactor + y * yfactor + x * xfactor;
                    auto offset_out =
                        z * the_velocities->shape[0] * the_velocities->shape[1] + y * the_velocities->shape[0] + x;
                    auto vx = VXVYVZ[offset_in * 3 + 0];
                    auto vy = VXVYVZ[offset_in * 3 + 1];
                    auto vz = VXVYVZ[offset_in * 3 + 2];
                    auto vmag = std::sqrt(vx * vx + vy * vy + vz * vz);
                    VMAG[offset_out] = vmag;
                    min = std::min(min, vmag);
                    max = std::max(max, vmag);
                }
            }
        }

        this->mins[0] = min;
        this->maxes[0] = max;
        this->metadata.MinValues = this->mins.data();
        this->metadata.MaxValues = this->maxes.data();

        currentFrame = vdc->FrameID();
    }
    return true;
}

bool ADIOSFlexVolume::onGetData(core::Call& call) {
    auto* vdc = dynamic_cast<geocalls::VolumetricDataCall*>(&call);
    if (vdc == nullptr)
        return false;

    auto* cad = this->adiosSlot.CallAs<CallADIOSData>();
    if (cad == nullptr)
        return false;

    if (!(*cad)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[ADIOSFlexVolume] Error during GetHeader");
        return false;
    }
    auto ret = assertData(vdc, cad);

    vdc->SetData(VMAG.data(), 1);
    vdc->SetDataHash(cad->getDataHash());
    vdc->SetFrameID(currentFrame);
    vdc->SetMetadata(&this->metadata);

    return ret;
}


bool ADIOSFlexVolume::onGetExtents(core::Call& call) {
    auto* vdc = dynamic_cast<geocalls::VolumetricDataCall*>(&call);
    if (vdc == nullptr)
        return false;

    auto* cad = this->adiosSlot.CallAs<CallADIOSData>();
    if (cad == nullptr)
        return false;

    if (!(*cad)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[ADIOSFlexVolume] Error during GetHeader");
        return false;
    }

    cad->setFrameIDtoLoad(vdc->FrameID());

    if (!inquireMetaDataVariables(cad))
        return false;

    if (!(*cad)(0)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("[ADIOSFlexVolume] Error during GetData");
        return false;
    }

    if (this->flexBoxSlot.Param<core::param::FlexEnumParam>()->ValueString() != "undef") {
        const auto the_box =
            cad->getData(this->flexBoxSlot.Param<core::param::FlexEnumParam>()->ValueString())->GetAsFloat();
        bbox.Set(the_box[0], the_box[1], the_box[2], the_box[3], the_box[4], the_box[5]);
    } else {
        bbox.Set(0.0f, 0.0f, 0.0f, 10.0f, 10.0f, 10.0f);
    }

    this->metadata.MemLoc = RAM;
    this->metadata.NumberOfFrames = cad->getFrameCount();

    this->metadata.GridType = CARTESIAN;
    this->metadata.Components = 1;
    this->metadata.ScalarLength = 4;
    this->metadata.ScalarType = FLOATING_POINT;


    this->metadata.Origin[0] = bbox.Left();
    this->metadata.Origin[1] = bbox.Bottom();
    this->metadata.Origin[2] = bbox.Back();
    this->metadata.Extents[0] = bbox.Width();
    this->metadata.Extents[1] = bbox.Height();
    this->metadata.Extents[2] = bbox.Depth();

    if (this->flexVelocitySlot.Param<core::param::FlexEnumParam>()->ValueString() == "undef") {
        return false;
    }

    std::stringstream vel_shape_str(
        cad->getVarProperties(this->flexVelocitySlot.Param<core::param::FlexEnumParam>()->ValueString())["Shape"]);
    std::vector<size_t> vel_shape;
    while (vel_shape_str.good()) {
        std::string substr;
        std::getline(vel_shape_str, substr, ',');
        vel_shape.push_back(std::atoi(substr.c_str()));
    }
    for (auto x = 0; x < 3; ++x) {
        this->metadata.Resolution[x] = vel_shape[x];
        metadata.SliceDists[0] = new float[1];
        metadata.SliceDists[0][0] = metadata.Extents[0] / static_cast<float>(metadata.Resolution[0] - 1);
        metadata.SliceDists[1] = new float[1];
        metadata.SliceDists[1][0] = metadata.Extents[1] / static_cast<float>(metadata.Resolution[1] - 1);
        metadata.SliceDists[2] = new float[1];
        metadata.SliceDists[2][0] = metadata.Extents[2] / static_cast<float>(metadata.Resolution[2] - 1);
        this->metadata.IsUniform[x] = true;
    }
    //this->mins[0] = min;
    //this->maxes[0] = max;
    //this->metadata.MinValues = this->mins.data();
    //this->metadata.MaxValues = this->maxes.data();

    vdc->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
    vdc->AccessBoundingBoxes().SetObjectSpaceClipBox(bbox);
    vdc->SetFrameCount(std::max<size_t>(cad->getFrameCount(), 1));
    vdc->SetMetadata(&this->metadata);

    return true;
}


bool ADIOSFlexVolume::onStartAsync(core::Call& call) {
    return true;
}


bool ADIOSFlexVolume::onStopAsync(core::Call& call) {
    return true;
}


bool ADIOSFlexVolume::onTryGetData(core::Call& call) {
    return true;
}

} // end namespace adios
} // end namespace megamol
