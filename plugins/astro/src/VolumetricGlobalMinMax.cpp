/*
 * VolumetricGlobalMinMax.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * All rights reserved.
 */

#include "VolumetricGlobalMinMax.h"

#include "mmcore/utility/log/Log.h"

/*
 * megamol::astro::VolumetricGlobalMinMax::VolumetricGlobalMinMax
 */
megamol::astro::VolumetricGlobalMinMax::VolumetricGlobalMinMax()
        : Module()
        , hash(0)
        , slotVolumetricDataIn("volumetricDataIn", "Input slot for volumetric data")
        , slotVolumetricDataOut("volumetricDataOut", "Output slot for volumetric data") {
    // Publish the slots.
    this->slotVolumetricDataIn.SetCompatibleCall<geocalls::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->slotVolumetricDataIn);

    this->slotVolumetricDataOut.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_GET_DATA),
        &VolumetricGlobalMinMax::onGetData);
    this->slotVolumetricDataOut.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_GET_EXTENTS),
        &VolumetricGlobalMinMax::onGetExtents);
    this->slotVolumetricDataOut.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_GET_METADATA),
        &VolumetricGlobalMinMax::onGetMetadata);
    this->slotVolumetricDataOut.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_START_ASYNC),
        &VolumetricGlobalMinMax::onUnsupportedCallback);
    this->slotVolumetricDataOut.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_STOP_ASYNC),
        &VolumetricGlobalMinMax::onUnsupportedCallback);
    this->slotVolumetricDataOut.SetCallback(geocalls::VolumetricDataCall::ClassName(),
        geocalls::VolumetricDataCall::FunctionName(geocalls::VolumetricDataCall::IDX_TRY_GET_DATA),
        &VolumetricGlobalMinMax::onUnsupportedCallback);
    this->MakeSlotAvailable(&this->slotVolumetricDataOut);
}

/*
 * megamol::astro::VolumetricGlobalMinMax::~VolumetricGlobalMinMax
 */
megamol::astro::VolumetricGlobalMinMax::~VolumetricGlobalMinMax() {
    this->Release();
}

/*
 * megamol::astro::VolumetricGlobalMinMax::create
 */
bool megamol::astro::VolumetricGlobalMinMax::create() {
    return true;
}

/*
 * megamol::astro::VolumetricGlobalMinMax::release
 */
void megamol::astro::VolumetricGlobalMinMax::release() {}

bool megamol::astro::VolumetricGlobalMinMax::onGetData(megamol::core::Call& call) {
    return pipeVolumetricDataCall(call, geocalls::VolumetricDataCall::IDX_GET_DATA);
}

bool megamol::astro::VolumetricGlobalMinMax::onGetExtents(megamol::core::Call& call) {
    return pipeVolumetricDataCall(call, geocalls::VolumetricDataCall::IDX_GET_EXTENTS);
}

bool megamol::astro::VolumetricGlobalMinMax::onGetMetadata(megamol::core::Call& call) {
    return pipeVolumetricDataCall(call, geocalls::VolumetricDataCall::IDX_GET_METADATA);
}

bool megamol::astro::VolumetricGlobalMinMax::onUnsupportedCallback(megamol::core::Call& call) {
    return false;
}

bool megamol::astro::VolumetricGlobalMinMax::pipeVolumetricDataCall(megamol::core::Call& call, unsigned int funcIdx) {
    using geocalls::VolumetricDataCall;
    using megamol::core::utility::log::Log;

    auto dst = dynamic_cast<VolumetricDataCall*>(&call);
    auto src = this->slotVolumetricDataIn.CallAs<VolumetricDataCall>();

    if (dst == nullptr) {
        Log::DefaultLog.WriteError("Call %hs of %hs received a wrong request.",
            VolumetricDataCall::FunctionName(funcIdx), VolumetricGlobalMinMax::ClassName());
        return false;
    }
    if (src == nullptr) {
        Log::DefaultLog.WriteError("Call %hs of %hs has a wrong source.", VolumetricDataCall::FunctionName(funcIdx),
            VolumetricGlobalMinMax::ClassName());
        return false;
    }

    *src = *dst;
    if (!(*src)(funcIdx)) {
        Log::DefaultLog.WriteError(
            "%hs failed to call %hs.", VolumetricGlobalMinMax::ClassName(), VolumetricDataCall::FunctionName(funcIdx));

        return false;
    }
    *dst = *src;

    if (src->DataHash() != this->hash || this->hash == 0) {
        this->hash = src->DataHash();
        this->minValues.clear();
        this->maxValues.clear();

        auto frames = src->FrameCount();
        for (unsigned int i = 0; i < frames; ++i) {
            src->SetFrameID(i, true);
            VolumetricDataCall::GetMetadata(*src);
            const auto metadata = src->GetMetadata();
            if (i == 0) {
                this->minValues.resize(metadata->Components, std::numeric_limits<double>::max());
                this->maxValues.resize(metadata->Components, std::numeric_limits<double>::lowest());
            } else {
                if (this->minValues.size() != metadata->Components) {
                    Log::DefaultLog.WriteError("Unexpected number of components.");
                    return false;
                }
            }

            for (size_t j = 0; j < metadata->Components; ++j) {
                if (metadata->MinValues[j] < this->minValues[j]) {
                    this->minValues[j] = metadata->MinValues[j];
                }
                if (metadata->MaxValues[j] > this->maxValues[j]) {
                    this->maxValues[j] = metadata->MaxValues[j];
                }
            }
        }
        Log::DefaultLog.WriteInfo("Min/Max Update");
        Log::DefaultLog.WriteInfo("Min:");
        for (const auto& m : this->minValues) {
            Log::DefaultLog.WriteInfo("    %f", m);
        }
        Log::DefaultLog.WriteInfo("Max:");
        for (const auto& m : this->maxValues) {
            Log::DefaultLog.WriteInfo("    %f", m);
        }
    }

    auto metadata = dst->GetMetadata();
    if (metadata != nullptr) {
        if (this->minValues.size() != metadata->Components) {
            Log::DefaultLog.WriteError("Unexpected number of components.");
            return false;
        }
        for (size_t i = 0; i < this->minValues.size(); ++i) {
            metadata->MinValues[i] = this->minValues[i];
            metadata->MaxValues[i] = this->maxValues[i];
        }
    }

    return true;
}
