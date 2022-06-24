/*
 * DifferenceVolume.cpp
 *
 * Copyright (C) 2019 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle rechte vorbehalten.
 */

#include "DifferenceVolume.h"

#include <limits>

#include "mmcore/param/BoolParam.h"

#include "mmcore/utility/log/Log.h"


/*
 * megamol::volume::DifferenceVolume::DifferenceVolume
 */
megamol::volume::DifferenceVolume::DifferenceVolume(void)
        : frameID((std::numeric_limits<unsigned int>::max)())
        , frameIdx(0)
        , hashData((std::numeric_limits<std::size_t>::max)())
        , hashState((std::numeric_limits<std::size_t>::max)())
        , paramIgnoreInputHash(
              "ignoreInputHash", "Instructs the module not to honour the input hash when checking for updates.")
        , slotIn("in", "The input slot providing the volume data.")
        , slotOut("out", "The output slot receiving the difference.") {
    using geocalls::VolumetricDataCall;

    this->slotIn.SetCompatibleCall<core::factories::CallAutoDescription<VolumetricDataCall>>();
    this->MakeSlotAvailable(&this->slotIn);

    this->slotOut.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA), &DifferenceVolume::onGetData);
    this->slotOut.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_EXTENTS), &DifferenceVolume::onGetExtents);
    this->slotOut.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_METADATA), &DifferenceVolume::onGetMetadata);
    this->slotOut.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_START_ASYNC), &DifferenceVolume::onUnsupported);
    this->slotOut.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_STOP_ASYNC), &DifferenceVolume::onUnsupported);
    this->slotOut.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_TRY_GET_DATA), &DifferenceVolume::onUnsupported);
    this->MakeSlotAvailable(&this->slotOut);

    this->paramIgnoreInputHash << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->paramIgnoreInputHash);
}


/*
 * megamol::volume::DifferenceVolume::~DifferenceVolume
 */
megamol::volume::DifferenceVolume::~DifferenceVolume(void) {
    this->Release();
}


/*
 * megamol::volume::DifferenceVolume::getFrameSize
 */
std::size_t megamol::volume::DifferenceVolume::getFrameSize(const geocalls::VolumetricMetadata_t& md) {
    auto retval = md.Resolution[0] * md.Resolution[1] * md.Resolution[2];
    retval *= md.ScalarLength;
    retval *= md.Components;
    return retval;
}


/*
 * megamol::volume::DifferenceVolume::getDifferenceType
 */
megamol::geocalls::ScalarType_t megamol::volume::DifferenceVolume::getDifferenceType(
    const geocalls::VolumetricMetadata_t& md) {
    switch (md.ScalarType) {
    case geocalls::SIGNED_INTEGER:
    case geocalls::FLOATING_POINT:
        // Can be used as it is.
        return md.ScalarType;

    case geocalls::UNSIGNED_INTEGER:
        // unsigned must become signed.
        return geocalls::SIGNED_INTEGER;

    default:
        // Anything else is unsupported.
        return geocalls::UNKNOWN;
    }
}


/*
 * megamol::volume::DifferenceVolume::checkCompatibility
 */
bool megamol::volume::DifferenceVolume::checkCompatibility(const geocalls::VolumetricMetadata_t& md) const {
    using megamol::core::utility::log::Log;
    auto reqType = getDifferenceType(md);

    if (getFrameSize(this->metadata) != getFrameSize(md)) {
        Log::DefaultLog.WriteError("The volume resolution must not change "
                                   "over time in order for %hs to work.",
            DifferenceVolume::ClassName());
        return false;
    }

    if ((this->metadata.ScalarLength != md.ScalarLength) && (md.ScalarType == reqType)) {
        Log::DefaultLog.WriteError("The scalar size must not change over time "
                                   "in order for %hs to work.",
            DifferenceVolume::ClassName());
        return false;
    }

    if (this->metadata.ScalarType != reqType) {
        Log::DefaultLog.WriteError("The scalar type must not change over time "
                                   "in order for %hs to work.",
            DifferenceVolume::ClassName());
        return false;
    }

    return true;
}


/*
 * megamol::volume::DifferenceVolume::create
 */
bool megamol::volume::DifferenceVolume::create(void) {
    return true;
}


/*
 * megamol::volume::DifferenceVolume::onGetData
 */
bool megamol::volume::DifferenceVolume::onGetData(core::Call& call) {
    using core::param::BoolParam;
    using geocalls::VolumetricDataCall;
    using megamol::core::utility::log::Log;

    auto dst = dynamic_cast<VolumetricDataCall*>(&call);
    auto src = this->slotIn.CallAs<VolumetricDataCall>();
    const auto localUpdate = this->paramIgnoreInputHash.IsDirty();
    const auto ignoreHash = this->paramIgnoreInputHash.Param<BoolParam>()->Value();

    /* Sanity checks. */
    if (dst == nullptr) {
        Log::DefaultLog.WriteError("Call %hs of %hs received a wrong request.",
            VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA), DifferenceVolume::ClassName());
        return false;
    }

    if (src == nullptr) {
        Log::DefaultLog.WriteError("Call %hs of %hs has a wrong source.",
            VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA), DifferenceVolume::ClassName());
        return false;
    }

    /* Retrieve info about incoming data and pass it on to the caller. */
    *src = *dst;
    src->SetFrameID(dst->FrameID(), true); // Use the force!
    assert(src->IsFrameForced());
    if (!VolumetricDataCall::GetMetadata(*src)) {
        return false;
    }
    *dst = *src;

    /* Establish what acceptable data are if the source changed. */
    if (localUpdate || (!ignoreHash && (this->hashData != src->DataHash()))) {
        Log::DefaultLog.WriteInfo("Volume data or local configuration have "
                                  "changed, resetting reference for difference computation.");

        /* Check for compatibility of the incoming data. */
        if (src->GetMetadata()->GridType != geocalls::CARTESIAN) {
            Log::DefaultLog.WriteError("%hs is only supported for Cartesian "
                                       "grids.",
                DifferenceVolume::ClassName());
            return false;
        }

        switch (src->GetMetadata()->ScalarType) {
        case geocalls::SIGNED_INTEGER:
        case geocalls::UNSIGNED_INTEGER:
        case geocalls::FLOATING_POINT:
            break;

        default:
            Log::DefaultLog.WriteError("%hs is not supported for scalar "
                                       "type %u.",
                DifferenceVolume::ClassName(), src->GetMetadata()->ScalarType);
            return false;
        }

        /* Everything is OK at this point, save the reference data. */
        this->metadata = *src->GetMetadata();

        /* Reset the caching state. */
        this->frameID = (std::numeric_limits<unsigned int>::max)();
        this->frameIdx = 0;
        this->hashData = src->DataHash();

        /* Mark local state as unchanged. */
        if (localUpdate) {
            this->paramIgnoreInputHash.ResetDirty();
        }
    }

    /* If the data we have need an update, compute it. */
    if (this->frameID != src->FrameID()) {
        Log::DefaultLog.WriteInfo("%hs is rebuilding the volume.", DifferenceVolume::ClassName());

        if (!VolumetricDataCall::GetMetadata(*src)) {
            return false;
        }

        /* Check that the format is the same. */
        if (!this->checkCompatibility(*src->GetMetadata())) {
            return false;
        }

        if (!(*src)(VolumetricDataCall::IDX_GET_DATA)) {
            Log::DefaultLog.WriteError("%hs failed to call %hs.", DifferenceVolume::ClassName(),
                VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA));
            return false;
        }

        /* Prepare a cache location for the current frame. */
        auto& cur = this->cache[this->frameIdx];
        cur.resize(src->GetFrameSize());
        ::memcpy(cur.data(), src->GetData(), src->GetFrameSize());

        /* Select the potential previous frame. */
        auto& prev = this->cache[increment(this->frameIdx)];

        /* Prepare the data storage. */
        this->data.resize(cur.size());

        if (src->FrameID() < 1) {
            /* There is no predecessor, so the frame is the difference. */
            Log::DefaultLog.WriteInfo("The data provided to %hs do not have a "
                                      "predecessor. The previous volume is considered to be zero.",
                DifferenceVolume::ClassName());
            auto& prev = this->cache[increment(this->frameIdx)];
            prev.resize(cur.size());
            ::memset(prev.data(), 0, prev.size());

        } else if (src->FrameID() - 1 != this->frameID) {
            /* We do not have the previous frame cached, so get it. */
            Log::DefaultLog.WriteInfo("Load previous frame %u to compute the "
                                      "difference to the current one.",
                src->FrameID() - 1);
            src->SetFrameID(src->FrameID() - 1, true);
            if (!VolumetricDataCall::GetMetadata(*src)) {
                return false;
            }

            /* Check that the format is the same. */
            if (!this->checkCompatibility(*src->GetMetadata())) {
                return false;
            }

            assert(src->IsFrameForced());
            if (!(*src)(VolumetricDataCall::IDX_GET_DATA)) {
                Log::DefaultLog.WriteError("%hs failed to call %hs.", DifferenceVolume::ClassName(),
                    VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA));
                return false;
            }

            prev.resize(src->GetFrameSize());
            ::memcpy(cur.data(), src->GetData(), src->GetFrameSize());
        }
        /* At this point, 'prev' contains the previous frame. */
        assert(cur.size() == this->data.size());
        assert(prev.size() == this->data.size());

        /* Do the conversion depending on the type of the data.*/
        switch (this->metadata.ScalarType) {
        case geocalls::SIGNED_INTEGER:
            switch (this->metadata.ScalarLength) {
            case 1: {
                auto c = reinterpret_cast<std::int8_t*>(cur.data());
                auto p = reinterpret_cast<std::int8_t*>(prev.data());
                auto d = reinterpret_cast<std::int8_t*>(this->data.data());
                this->calcDifference(d, c, p, cur.size() / sizeof(*c));
            } break;

            case 2: {
                auto c = reinterpret_cast<std::int16_t*>(cur.data());
                auto p = reinterpret_cast<std::int16_t*>(prev.data());
                auto d = reinterpret_cast<std::int16_t*>(this->data.data());
                this->calcDifference(d, c, p, cur.size() / sizeof(*c));
            } break;

            case 4: {
                auto c = reinterpret_cast<std::int32_t*>(cur.data());
                auto p = reinterpret_cast<std::int32_t*>(prev.data());
                auto d = reinterpret_cast<std::int32_t*>(this->data.data());
                this->calcDifference(d, c, p, cur.size() / sizeof(*c));
            } break;

            case 8: {
                auto c = reinterpret_cast<std::int64_t*>(cur.data());
                auto p = reinterpret_cast<std::int64_t*>(prev.data());
                auto d = reinterpret_cast<std::int64_t*>(this->data.data());
                this->calcDifference(d, c, p, cur.size() / sizeof(*c));
            } break;

            default:
                Log::DefaultLog.WriteError("%hs cannot process "
                                           "%u-byte SIGNED_INTEGER data.",
                    DifferenceVolume::ClassName(), this->metadata.ScalarLength);
                return false;
            }
            break;

        case geocalls::UNSIGNED_INTEGER:
            // HAZARD: unsigned-to-signed conversion is untested!
            switch (this->metadata.ScalarLength) {
            case 1: {
                auto c = (cur.data());
                auto p = (prev.data());
                this->data.resize(2 * this->data.size());
                this->metadata.ScalarLength = 2 * src->GetMetadata()->ScalarLength;
                auto d = reinterpret_cast<std::int16_t*>(this->data.data());
                this->calcDifference(d, c, p, cur.size() / sizeof(*c));
            } break;

            case 2: {
                auto c = reinterpret_cast<std::uint16_t*>(cur.data());
                auto p = reinterpret_cast<std::uint16_t*>(prev.data());
                this->data.resize(2 * this->data.size());
                this->metadata.ScalarLength = 2 * src->GetMetadata()->ScalarLength;
                auto d = reinterpret_cast<std::int32_t*>(this->data.data());
                this->calcDifference(d, c, p, cur.size() / sizeof(*c));
            } break;

            case 4: {
                auto c = reinterpret_cast<std::uint32_t*>(cur.data());
                auto p = reinterpret_cast<std::uint32_t*>(prev.data());
                this->data.resize(2 * this->data.size());
                this->metadata.ScalarLength = 2 * src->GetMetadata()->ScalarLength;
                auto d = reinterpret_cast<std::int64_t*>(this->data.data());
                this->calcDifference(d, c, p, cur.size() / sizeof(*c));
            } break;

            case 8: {
                Log::DefaultLog.WriteWarn("Conversion from UINT64 "
                                          "to INT64 in %hs might cause data truncation.",
                    DifferenceVolume::ClassName());
                auto c = reinterpret_cast<std::uint64_t*>(cur.data());
                auto p = reinterpret_cast<std::uint64_t*>(prev.data());
                auto d = reinterpret_cast<std::int64_t*>(this->data.data());
                this->calcDifference(d, c, p, cur.size() / sizeof(*c));
            } break;

            default:
                Log::DefaultLog.WriteError("%hs cannot process "
                                           "%u-byte UNSIGNED_INTEGER data.",
                    DifferenceVolume::ClassName(), this->metadata.ScalarLength);
                return false;
            }
            break;

        case geocalls::FLOATING_POINT:
            switch (this->metadata.ScalarLength) {
            case 4: {
                auto c = reinterpret_cast<float*>(cur.data());
                auto p = reinterpret_cast<float*>(prev.data());
                auto d = reinterpret_cast<float*>(this->data.data());
                this->calcDifference(d, c, p, cur.size() / sizeof(*c));
            } break;

            case 8: {
                auto c = reinterpret_cast<double*>(cur.data());
                auto p = reinterpret_cast<double*>(prev.data());
                auto d = reinterpret_cast<double*>(this->data.data());
                this->calcDifference(d, c, p, cur.size() / sizeof(*c));
            } break;

            default:
                Log::DefaultLog.WriteError("%hs cannot process "
                                           "%u-byte FLOATING_POINT data.",
                    DifferenceVolume::ClassName(), this->metadata.ScalarLength);
                return false;
            }
            break;

        default:
            assert(false); // This should not be reachable.
            return false;
        }

        this->frameID = src->FrameID();
        this->frameIdx = increment(this->frameIdx);
    } /* end if (this->frameID != src->FrameID()) */

    dst->SetData(this->data.data());
    dst->SetMetadata(&this->metadata);

    dst->SetDataHash(this->getHash());
    return true;
}


/*
 * megamol::volume::DifferenceVolume::onGetExtents
 */
bool megamol::volume::DifferenceVolume::onGetExtents(core::Call& call) {
    using geocalls::VolumetricDataCall;
    using megamol::core::utility::log::Log;

    auto dst = dynamic_cast<VolumetricDataCall*>(&call);
    auto src = this->slotIn.CallAs<VolumetricDataCall>();

    if (dst == nullptr) {
        Log::DefaultLog.WriteError("Call %hs of %hs received a wrong request.",
            VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_EXTENTS), DifferenceVolume::ClassName());
        return false;
    }

    if (src == nullptr) {
        Log::DefaultLog.WriteError("Call %hs of %hs has a wrong source.",
            VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_EXTENTS), DifferenceVolume::ClassName());
        return false;
    }

    *src = *dst;
    if (!(*src)(VolumetricDataCall::IDX_GET_EXTENTS)) {
        Log::DefaultLog.WriteError("%hs failed to call %hs.", DifferenceVolume::ClassName(),
            VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_EXTENTS));
        return false;
    }
    *dst = *src;

    dst->SetDataHash(this->getHash());
    return true;
}


/*
 * megamol::volume::DifferenceVolume::onGetMetadata
 */
bool megamol::volume::DifferenceVolume::onGetMetadata(core::Call& call) {
    using geocalls::VolumetricDataCall;
    using megamol::core::utility::log::Log;

    auto dst = dynamic_cast<VolumetricDataCall*>(&call);
    auto src = this->slotIn.CallAs<VolumetricDataCall>();

    if (dst == nullptr) {
        Log::DefaultLog.WriteError("Call %hs of %hs received a wrong request.",
            VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_METADATA), DifferenceVolume::ClassName());
        return false;
    }

    if (src == nullptr) {
        Log::DefaultLog.WriteError("Call %hs of %hs has a wrong source.",
            VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_METADATA), DifferenceVolume::ClassName());
        return false;
    }

    *src = *dst;
    if (!VolumetricDataCall::GetMetadata(*src)) {
        return false;
    }
    *dst = *src;

    dst->SetDataHash(this->getHash());
    return true;
}


/*
 * megamol::volume::DifferenceVolume::onUnsupported
 */
bool megamol::volume::DifferenceVolume::onUnsupported(core::Call& call) {
    return false;
}


/*
 * megamol::volume::DifferenceVolume::release
 */
void megamol::volume::DifferenceVolume::release(void) {
    for (auto& c : this->cache) {
        c.clear();
    }
    this->data.clear();
}
