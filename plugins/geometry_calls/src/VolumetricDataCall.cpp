/*
 * VolumetricDataCall.cpp
 *
 * Copyright (C) 2014 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle rechte vorbehalten.
 */

#include "geometry_calls/VolumetricDataCall.h"

#include <utility>

#include "vislib/OutOfRangeException.h"

#include "mmcore/utility/log/Log.h"


#define STATIC_ARRAY_COUNT(ary) (sizeof(ary) / sizeof(*(ary)))

namespace megamol::geocalls {

/*
 * megamol::pcl::CallPcd::FunctionCount
 */
unsigned int VolumetricDataCall::FunctionCount() {
    return STATIC_ARRAY_COUNT(VolumetricDataCall::FUNCTIONS);
}


/*
 * megamol::pcl::CallPcd::FunctionName
 */
const char* VolumetricDataCall::FunctionName(unsigned int idx) {
    if (idx < VolumetricDataCall::FunctionCount()) {
        return VolumetricDataCall::FUNCTIONS[idx];
    } else {
        return "";
    }
}


/*
 * VolumetricDataCall::GetMetadata
 */
bool VolumetricDataCall::GetMetadata(VolumetricDataCall& call) {
    using geocalls::VolumetricDataCall;
    using megamol::core::utility::log::Log;

    if (!call(VolumetricDataCall::IDX_GET_METADATA)) {
        Log::DefaultLog.WriteError("%hs::%hs failed.", VolumetricDataCall::ClassName(),
            VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_METADATA));
        return false;
    }

    if (call.GetMetadata() == nullptr) {
        /* Second chance ... */
        if (!call(VolumetricDataCall::IDX_GET_DATA)) {
            Log::DefaultLog.WriteError("%hs::%hs failed.", VolumetricDataCall::ClassName(),
                VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA));
            return false;
        }
    }

    auto retval = (call.GetMetadata() != nullptr);

    if (!retval) {
        Log::DefaultLog.WriteError("Call to %hs::%hs or %hs::%hs succeeded, "
                                   "but none of them did provide any metadata. The call will be "
                                   "considered to have failed.",
            VolumetricDataCall::ClassName(), VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_METADATA),
            VolumetricDataCall::ClassName(), VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA));
    }

    return retval;
}


/*
 * VolumetricDataCall::IDX_GET_DATA
 */
const unsigned int VolumetricDataCall::IDX_GET_DATA = 1;


/*
 * VolumetricDataCall::IDX_GET_EXTENTS
 */
const unsigned int VolumetricDataCall::IDX_GET_EXTENTS = 0;


/*
 * VolumetricDataCall::IDX_GET_METADATA
 */
const unsigned int VolumetricDataCall::IDX_GET_METADATA = 2;


/*
 * VolumetricDataCall::IDX_START_ASYNC
 */
const unsigned int VolumetricDataCall::IDX_START_ASYNC = 3;


/*
 * VolumetricDataCall::IDX_STOP_ASYNC
 */
const unsigned int VolumetricDataCall::IDX_STOP_ASYNC = 4;


/*
 * VolumetricDataCall::IDX_TRY_GET_DATA
 */
const unsigned int VolumetricDataCall::IDX_TRY_GET_DATA = 5;


/*
 * VolumetricDataCall::VolumetricDataCall
 */
VolumetricDataCall::VolumetricDataCall() : data(nullptr), metadata(nullptr), vram_volume_name(0) {}


/*
 * VolumetricDataCall::VolumetricDataCall
 */
VolumetricDataCall::VolumetricDataCall(const VolumetricDataCall& rhs)
        : data(nullptr)
        , metadata(nullptr)
        , vram_volume_name(0) {
    *this = rhs;
}


/*
 * VolumetricDataCall::~VolumetricDataCall
 */
VolumetricDataCall::~VolumetricDataCall() {}


/*
 * VolumetricDataCall::GetComponents
 */
size_t VolumetricDataCall::GetComponents() const {
    return (this->metadata != nullptr) ? this->metadata->Components : 0;
}


/*
 * VolumetricDataCall::GetFrames
 */
size_t VolumetricDataCall::GetFrames() const {
    return (this->metadata != nullptr) ? this->metadata->NumberOfFrames : 0;
}


/*
 * VolumetricDataCall::GetGridType
 */
VolumetricDataCall::GridType VolumetricDataCall::GetGridType() const {
    return (this->metadata != nullptr) ? this->metadata->GridType : GridType::NONE;
}


/*
 * VolumetricDataCall::GetResolution
 */
const size_t VolumetricDataCall::GetResolution(const int axis) const {
    if ((axis < 0) || (axis > 2)) {
        throw vislib::OutOfRangeException(axis, 0, 2, __FILE__, __LINE__);
    }
    return (this->metadata != nullptr) ? this->metadata->Resolution[axis] : 0;
}


/*
 * VolumetricDataCall::GetScalarLength
 */
size_t VolumetricDataCall::GetScalarLength() const {
    return (this->metadata != nullptr) ? this->metadata->ScalarLength : 0;
}


/*
 * VolumetricDataCall::GetScalarType
 */
VolumetricDataCall::ScalarType VolumetricDataCall::GetScalarType() const {
    return (this->metadata != nullptr) ? this->metadata->ScalarType : ScalarType::UNKNOWN;
}


/*
 * VolumetricDataCall::GetSliceDistances
 */
const float* VolumetricDataCall::GetSliceDistances(const int axis) const {
    if ((axis < 0) || (axis > 2)) {
        throw vislib::OutOfRangeException(axis, 0, 2, __FILE__, __LINE__);
    }
    return (this->metadata != nullptr) ? this->metadata->SliceDists[axis] : nullptr;
}


/*
 * VolumetricDataCall::GetVoxelsPerFrame
 */
size_t VolumetricDataCall::GetVoxelsPerFrame() const {
    if (this->metadata == nullptr) {
        return 0;
    } else {
        size_t retval = 1;
        for (int i = 0; i < STATIC_ARRAY_COUNT(this->metadata->Resolution); ++i) {
            retval *= this->metadata->Resolution[i];
        }
        return retval;
    }
}


const float VolumetricDataCall::GetRelativeVoxelValue(
    const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t c) const {

    float theVal = 0.0f;

    if (!this->metadata->IsUniform || !this->metadata->GridType == GridType::CARTESIAN ||
        !this->metadata->GridType == GridType::RECTILINEAR) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("GetRelativeVoxelValue: unsupported grid!");

    } else {
        uint64_t idx = (z * this->metadata->Resolution[1] + y) * this->metadata->Resolution[0] + x;
        idx *= this->metadata->Components;
        switch (this->metadata->ScalarType) {
        case UNKNOWN:
        case BITS:
            megamol::core::utility::log::Log::DefaultLog.WriteError("GetRelativeVoxelValue: unsupported scalar type!");
            break;

        case SIGNED_INTEGER: {
            const int* theVol = static_cast<int*>(this->data);
            theVal = theVol[idx];
            theVal -= this->metadata->MinValues[c];
            theVal /= (this->metadata->MaxValues[c] - this->metadata->MinValues[c]);
        } break;

        case UNSIGNED_INTEGER: {
            const unsigned int* theVol = static_cast<unsigned int*>(this->data);
            theVal = theVol[idx];
            theVal -= this->metadata->MinValues[c];
            theVal /= (this->metadata->MaxValues[c] - this->metadata->MinValues[c]);
        } break;

        case FLOATING_POINT: {
            const float* theVol = static_cast<float*>(this->data);
            theVal = theVol[idx];
            theVal -= this->metadata->MinValues[c];
            theVal /= (this->metadata->MaxValues[c] - this->metadata->MinValues[c]);
        } break;
        }
    }

    return theVal;
}

const float VolumetricDataCall::GetAbsoluteVoxelValue(
    const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t c) const {

    float theVal = 0.0f;

    if (!this->metadata->IsUniform || !this->metadata->GridType == GridType::CARTESIAN ||
        !this->metadata->GridType == GridType::RECTILINEAR) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("GetAbsoluteVoxelValue: unsupported grid!");

    } else {
        uint64_t idx =
            z * this->metadata->Resolution[0] * this->metadata->Resolution[1] + y * this->metadata->Resolution[0] + x;
        idx *= this->metadata->Components;
        assert(idx < metadata->Resolution[0] * metadata->Resolution[1] * metadata->Resolution[2]);
        switch (this->metadata->ScalarType) {
        case UNKNOWN:
        case BITS:
            megamol::core::utility::log::Log::DefaultLog.WriteError("GetAbsoluteVoxelValue: unsupported scalar type!");
            break;

        case SIGNED_INTEGER: {
            const int* theVol = static_cast<int*>(this->data);
            theVal = theVol[idx];
        } break;

        case UNSIGNED_INTEGER: {
            const unsigned int* theVol = static_cast<unsigned int*>(this->data);
            theVal = theVol[idx];
        } break;

        case FLOATING_POINT: {
            const float* theVol = static_cast<float*>(this->data);
            theVal = theVol[idx];
        } break;
        }
    }

    return theVal;
}

/*
 * VolumetricDataCall::IsUniform
 */
bool VolumetricDataCall::IsUniform(const int axis) const {
    if ((axis < 0) || (axis > 2)) {
        throw vislib::OutOfRangeException(axis, 0, 2, __FILE__, __LINE__);
    }
    return (this->metadata != nullptr) ? this->metadata->IsUniform[axis] : true;
}


/*
 * VolumetricDataCall::SetMetadata
 */
void VolumetricDataCall::SetMetadata(const Metadata* metadata) {
    this->metadata = metadata;
}


/*
 * VolumetricDataCall::operator =
 */
VolumetricDataCall& VolumetricDataCall::operator=(const VolumetricDataCall& rhs) {
    if (this != &rhs) {
        Base::operator=(rhs);
        this->data = rhs.data;
        this->metadata = rhs.metadata;
    }
    return *this;
}


/*
 * VolumetricDataCall::FUNCTIONS
 */
const char* VolumetricDataCall::FUNCTIONS[] = {
    "GetExtents", "GetData", "GetMetadata", "StartAsync", "StopAsync", "TryGetData"};
} // namespace megamol::geocalls
