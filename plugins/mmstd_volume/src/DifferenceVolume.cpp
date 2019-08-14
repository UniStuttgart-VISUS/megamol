/*
 * DifferenceVolume.cpp
 *
 * Copyright (C) 2019 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle rechte vorbehalten.
 */

#include "stdafx.h"
#include "DifferenceVolume.h"

#include <limits>

#include "mmcore/param/EnumParam.h"

#include "vislib/sys/Log.h"


/*
 * megamol::stdplugin::volume::DifferenceVolume::DifferenceVolume
 */
megamol::stdplugin::volume::DifferenceVolume::DifferenceVolume(void)
        : frameID((std::numeric_limits<unsigned int>::max)()),
        frameIdx(0),
        hashData((std::numeric_limits<std::size_t>::max)()),
        hashState((std::numeric_limits<std::size_t>::max)()),
        slotIn("in", "The input slot providing the volume data."),
        slotOut("out", "The output slot receiving the difference.") {
    using core::misc::VolumetricDataCall;

    this->slotIn.SetCompatibleCall<
        core::factories::CallAutoDescription<VolumetricDataCall>>();
    this->MakeSlotAvailable(&this->slotIn);

    this->slotOut.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA),
        &DifferenceVolume::onGetData);
    this->slotOut.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_EXTENTS),
        &DifferenceVolume::onGetExtents);
    this->slotOut.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_METADATA),
        &DifferenceVolume::onGetMetadata);
    this->slotOut.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_START_ASYNC),
        &DifferenceVolume::onUnsupported);
    this->slotOut.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_STOP_ASYNC),
        &DifferenceVolume::onUnsupported);
    this->slotOut.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_TRY_GET_DATA),
        &DifferenceVolume::onUnsupported);
    this->MakeSlotAvailable(&this->slotOut);
}


/*
 * megamol::stdplugin::volume::DifferenceVolume::~DifferenceVolume
 */
megamol::stdplugin::volume::DifferenceVolume::~DifferenceVolume(void) {
    this->Release();
}


/*
 * megamol::stdplugin::volume::DifferenceVolume::getFrameSize
 */
std::size_t megamol::stdplugin::volume::DifferenceVolume::getFrameSize(
        const core::misc::VolumetricMetadata_t& md) {
    auto retval = md.Resolution[0] * md.Resolution[1] * md.Resolution[2];
    retval *= md.ScalarLength;
    retval *= md.Components;
    return retval;
}


/*
 * megamol::stdplugin::volume::DifferenceVolume::checkCompatibility
 */
bool megamol::stdplugin::volume::DifferenceVolume::checkCompatibility(
        const core::misc::VolumetricMetadata_t& md) const {
    using vislib::sys::Log;

    if (getFrameSize(this->metadata) != getFrameSize(md)) {
        Log::DefaultLog.WriteError(L"The volume resolution must not change "
            "over time in order for %hs to work.",
            DifferenceVolume::ClassName());
        return false;
    }

    if (this->metadata.ScalarLength != md.ScalarLength) {
        Log::DefaultLog.WriteError(L"The scalar size must not change over time "
            "in order for %hs to work.",
            DifferenceVolume::ClassName());
        return false;
    }

    if (this->metadata.ScalarType != md.ScalarType) {
        Log::DefaultLog.WriteError(L"The scalar type must not change over time "
            "in order for %hs to work.",
            DifferenceVolume::ClassName());
        return false;
    }

    return true;
}


/*
 * megamol::stdplugin::volume::DifferenceVolume::create
 */
bool megamol::stdplugin::volume::DifferenceVolume::create(void) {
    return true;
}


/*
 * megamol::stdplugin::volume::DifferenceVolume::onGetData
 */
bool megamol::stdplugin::volume::DifferenceVolume::onGetData(core::Call& call) {
    using core::misc::VolumetricDataCall;
    using vislib::sys::Log;

    auto dst = dynamic_cast<VolumetricDataCall *>(&call);
    auto src = this->slotIn.CallAs<VolumetricDataCall>();

    /* Sanity checks. */
    if (dst == nullptr) {
        Log::DefaultLog.WriteError(L"Call %hs of %hs received a wrong request.",
            VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA),
            DifferenceVolume::ClassName());
        return false;
    }

    if (src == nullptr) {
        Log::DefaultLog.WriteError(L"Call %hs of %hs has a wrong source.",
            VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA),
            DifferenceVolume::ClassName());
        return false;
    }

    /* Retrieve info about incoming data and pass it on to the caller. */
    *src = *dst;
    if (!VolumetricDataCall::GetMetadata(*src)) {
        return false;
    }
    *dst = *src;

    /* Establish what acceptable data are if the source changed. */
    if (this->hashData != src->DataHash()) {
        Log::DefaultLog.WriteInfo(L"Volume data has changed, resetting "
            L"reference for difference computation.");

        /* Check for compatibility of the incoming data. */
        if (src->GetMetadata()->GridType != core::misc::CARTESIAN) {
            Log::DefaultLog.WriteError(L"%hs is only supported for Cartesian "
                L"grids.", DifferenceVolume::ClassName());
            return false;
        }

        switch (src->GetMetadata()->ScalarType) {
            case core::misc::SIGNED_INTEGER:
            case core::misc::UNSIGNED_INTEGER:
            case core::misc::FLOATING_POINT:
                break;

            default:
                Log::DefaultLog.WriteError(L"%hs is not supported for scalar "
                    "type %u.", DifferenceVolume::ClassName(),
                    src->GetMetadata()->ScalarType);
                return false;
        }

        /* Everything is OK at this point, save the reference data. */
        this->metadata = *src->GetMetadata();

        /* Reset the caching state. */
        this->frameID = (std::numeric_limits<unsigned int>::max)();
        this->frameIdx = 0;
        this->hashData = src->DataHash();
    }

    /* If the data we have need an update, compute it. */
    if (this->frameID != src->FrameID()) {
        if (!VolumetricDataCall::GetMetadata(*src)) {
            return false;
        }

        /* Check that the format is the same. */
        if (!this->checkCompatibility(*src->GetMetadata())) {
            return false;
        }

        if (!(*src)(VolumetricDataCall::IDX_GET_DATA)) {
            Log::DefaultLog.WriteError(L"%hs failed to call %hs.",
                DifferenceVolume::ClassName(),
                VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA));
            return false;
        }

        /* Prepare a cache location for the current frame. */
        auto& cur = this->cache[this->frameIdx];
        cur.resize(src->GetFrameSize());
        ::memcpy(cur.data(), src->GetData(), src->GetFrameSize());

        /* Prepare the data storage. */
        this->data.resize(cur.size());

        if (src->FrameID() < 1) {
            /* There is no predecessor, so the frame is the difference. */
            Log::DefaultLog.WriteInfo(L"The data provided to %hs do not have a "
                L"predecessor. The previous volume is considered to be zero.",
                DifferenceVolume::ClassName());
            std::copy(cur.begin(), cur.end(), this->data.begin());

        } else {
            /* Compute difference to previous frame. */
            auto& prev = this->cache[increment(this->frameIdx)];

            if (src->FrameID() - 1 != this->frameID) {
                /* We do not have the previous frame cached, so get it. */

                src->SetFrameID(src->FrameID() - 1);
                if (!VolumetricDataCall::GetMetadata(*src)) {
                    return false;
                }

                /* Check that the format is the same. */
                if (!this->checkCompatibility(*src->GetMetadata())) {
                    return false;
                }

                if (!(*src)(VolumetricDataCall::IDX_GET_DATA)) {
                    Log::DefaultLog.WriteError(L"%hs failed to call %hs.",
                        DifferenceVolume::ClassName(),
                        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA));
                    return false;
                }

                prev.resize(src->GetFrameSize());
                ::memcpy(cur.data(), src->GetData(), src->GetFrameSize());
            }
            /* At this point, 'prev' contains the previous frame. */
            assert(cur.size() == this->data.size());
            assert(prev.size() == this->data.size());

            switch (this->metadata.ScalarType) {
                case core::misc::SIGNED_INTEGER:
                    switch (this->metadata.ScalarLength) {
                        case 1: {
                            auto c = reinterpret_cast<std::int8_t *>(cur.data());
                            auto p = reinterpret_cast<std::int8_t *>(prev.data());
                            auto d = reinterpret_cast<std::int8_t *>(this->data.data());
#pragma omp parallel for
                            for (std::size_t i = 0; i < cur.size() / sizeof(*c); ++i) {
                                d[i] = c[i] - p[i];
                            }
                            } break;

                        case 2: {
                            auto c = reinterpret_cast<std::int16_t *>(cur.data());
                            auto p = reinterpret_cast<std::int16_t *>(prev.data());
                            auto d = reinterpret_cast<std::int16_t *>(this->data.data());
#pragma omp parallel for
                            for (std::size_t i = 0; i < cur.size() / sizeof(*c); ++i) {
                                d[i] = c[i] - p[i];
                            }
                            } break;

                        case 4: {
                            auto c = reinterpret_cast<std::int32_t *>(cur.data());
                            auto p = reinterpret_cast<std::int32_t *>(prev.data());
                            auto d = reinterpret_cast<std::int32_t *>(this->data.data());
#pragma omp parallel for
                            for (std::size_t i = 0; i < cur.size() / sizeof(*c); ++i) {
                                d[i] = c[i] - p[i];
                            }
                            } break;

                        case 8: {
                            auto c = reinterpret_cast<std::int64_t *>(cur.data());
                            auto p = reinterpret_cast<std::int64_t *>(prev.data());
                            auto d = reinterpret_cast<std::int64_t *>(this->data.data());
#pragma omp parallel for
                            for (std::size_t i = 0; i < cur.size() / sizeof(*c); ++i) {
                                d[i] = c[i] - p[i];
                            }
                            } break;

                        default:
                            Log::DefaultLog.WriteError(L"%hs cannot process "
                                L"%u-byte SIGNED_INTEGER data.",
                                DifferenceVolume::ClassName(),
                                this->metadata.ScalarLength);
                            return false;
                    }
                    break;

#if 0
                    // TOOD: THIS IS BS! We need to expand the data type
                case core::misc::UNSIGNED_INTEGER:
                    switch (this->scalarSize) {
                        case 1: {
                            auto c = (cur.data());
                            auto p = (prev.data());
                            auto d = (this->data.data());
#pragma omp parallel for
                            for (std::size_t i = 0; i < cur.size() / sizeof(*c); ++i) {
                                d[i] = c[i] - p[i];
                            }
                            } break;

                        case 2: {
                            auto c = reinterpret_cast<std::uint16_t *>(cur.data());
                            auto p = reinterpret_cast<std::uint16_t *>(prev.data());
                            auto d = reinterpret_cast<std::uint16_t *>(this->data.data());
#pragma omp parallel for
                            for (std::size_t i = 0; i < cur.size() / sizeof(*c); ++i) {
                                d[i] = c[i] - p[i];
                            }
                            } break;

                        case 4: {
                            auto c = reinterpret_cast<std::uint32_t *>(cur.data());
                            auto p = reinterpret_cast<std::uint32_t *>(prev.data());
                            auto d = reinterpret_cast<std::uint32_t *>(this->data.data());
#pragma omp parallel for
                            for (std::size_t i = 0; i < cur.size() / sizeof(*c); ++i) {
                                d[i] = c[i] - p[i];
                            }
                            } break;

                        case 8: {
                            auto c = reinterpret_cast<std::uint64_t *>(cur.data());
                            auto p = reinterpret_cast<std::uint64_t *>(prev.data());
                            auto d = reinterpret_cast<std::uint64_t *>(this->data.data());
#pragma omp parallel for
                            for (std::size_t i = 0; i < cur.size() / sizeof(*c); ++i) {
                                d[i] = c[i] - p[i];
                            }
                            } break;

                        default:
                            Log::DefaultLog.WriteError(L"%hs cannot process "
                                L"%u-byte UNSIGNED_INTEGER data.",
                                DifferenceVolume::ClassName(),
                                this->scalarSize);
                            return false;
                    }
                    break;
#endif

                case core::misc::FLOATING_POINT:
                    switch (this->metadata.ScalarLength) {
                        case 4: {
                            auto c = reinterpret_cast<float *>(cur.data());
                            auto p = reinterpret_cast<float *>(prev.data());
                            auto d = reinterpret_cast<float *>(this->data.data());
#pragma omp parallel for
                            for (std::size_t i = 0; i < cur.size() / sizeof(*c); ++i) {
                                d[i] = c[i] - p[i];
                            }
                            } break;

                        case 8: {
                            auto c = reinterpret_cast<double *>(cur.data());
                            auto p = reinterpret_cast<double *>(prev.data());
                            auto d = reinterpret_cast<double *>(this->data.data());
#pragma omp parallel for
                            for (std::size_t i = 0; i < cur.size() / sizeof(*c); ++i) {
                                d[i] = c[i] - p[i];
                            }
                            } break;

                        default:
                            Log::DefaultLog.WriteError(L"%hs cannot process "
                                L"%u-byte FLOATING_POINT data.",
                                DifferenceVolume::ClassName(),
                                this->metadata.ScalarLength);
                            return false;
                    }
                    break;

                default:
                    assert(false);  // This should not be reachable.
                    return false;
            }
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
 * megamol::stdplugin::volume::DifferenceVolume::onGetExtents
 */
bool megamol::stdplugin::volume::DifferenceVolume::onGetExtents(core::Call& call) {
    using core::misc::VolumetricDataCall;
    using vislib::sys::Log;

    auto dst = dynamic_cast<VolumetricDataCall *>(&call);
    auto src = this->slotIn.CallAs<VolumetricDataCall>();

    if (dst == nullptr) {
        Log::DefaultLog.WriteError(L"Call %hs of %hs received a wrong request.",
            VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_EXTENTS),
            DifferenceVolume::ClassName());
        return false;
    }

    if (src == nullptr) {
        Log::DefaultLog.WriteError(L"Call %hs of %hs has a wrong source.",
            VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_EXTENTS),
            DifferenceVolume::ClassName());
        return false;
    }

    *src = *dst;
    if (!(*src)(VolumetricDataCall::IDX_GET_EXTENTS)) {
        Log::DefaultLog.WriteError(L"%hs failed to call %hs.",
            DifferenceVolume::ClassName(),
            VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_EXTENTS));
        return false;
    }
    *dst = *src;

    dst->SetDataHash(this->getHash());
    return true;
}


/*
 * megamol::stdplugin::volume::DifferenceVolume::onGetMetadata
 */
bool megamol::stdplugin::volume::DifferenceVolume::onGetMetadata(
        core::Call& call) {
    using core::misc::VolumetricDataCall;
    using vislib::sys::Log;

    auto dst = dynamic_cast<VolumetricDataCall *>(&call);
    auto src = this->slotIn.CallAs<VolumetricDataCall>();

    if (dst == nullptr) {
        Log::DefaultLog.WriteError(L"Call %hs of %hs received a wrong request.",
            VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_METADATA),
            DifferenceVolume::ClassName());
        return false;
    }

    if (src == nullptr) {
        Log::DefaultLog.WriteError(L"Call %hs of %hs has a wrong source.",
            VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_METADATA),
            DifferenceVolume::ClassName());
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
 * megamol::stdplugin::volume::DifferenceVolume::onUnsupported
 */
bool megamol::stdplugin::volume::DifferenceVolume::onUnsupported(
        core::Call& call) {
    return false;
}


/*
 * megamol::stdplugin::volume::DifferenceVolume::release
 */
void megamol::stdplugin::volume::DifferenceVolume::release(void) {
    for (auto& c : this->cache) {
        c.clear();
    }
    this->data.clear();
}
