/*
 * VolumetricDataCall.cpp
 *
 * Copyright (C) 2014 by Visualisierungsinstitut der Universit�t Stuttgart.
 * Alle rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/misc/VolumetricDataCall.h"

#include <utility>

#include "vislib/OutOfRangeException.h"


#define STATIC_ARRAY_COUNT(ary) (sizeof(ary) / sizeof(*(ary)))


/*
 * megamol::pcl::CallPcd::FunctionCount
 */
unsigned int megamol::core::misc::VolumetricDataCall::FunctionCount(void) {
    return STATIC_ARRAY_COUNT(VolumetricDataCall::FUNCTIONS);
}


/*
 * megamol::pcl::CallPcd::FunctionName
 */
const char *megamol::core::misc::VolumetricDataCall::FunctionName(
        unsigned int idx) {
    if (idx < VolumetricDataCall::FunctionCount()) {
        return VolumetricDataCall::FUNCTIONS[idx];
    } else {
        return "";
    }
}


/*
 * megamol::core::misc::VolumetricDataCall::IDX_GET_DATA
 */
const unsigned int megamol::core::misc::VolumetricDataCall::IDX_GET_DATA = 1;


/*
 * megamol::core::misc::VolumetricDataCall::IDX_GET_EXTENTS
 */
const unsigned int megamol::core::misc::VolumetricDataCall::IDX_GET_EXTENTS = 0;


/*
 * megamol::core::misc::VolumetricDataCall::IDX_GET_METADATA
 */
const unsigned int megamol::core::misc::VolumetricDataCall::IDX_GET_METADATA = 2;


/*
 * megamol::core::misc::VolumetricDataCall::IDX_START_ASYNC
 */
const unsigned int megamol::core::misc::VolumetricDataCall::IDX_START_ASYNC = 3;


/*
 * megamol::core::misc::VolumetricDataCall::IDX_STOP_ASYNC
 */
const unsigned int megamol::core::misc::VolumetricDataCall::IDX_STOP_ASYNC = 4;


/*
 * megamol::core::misc::VolumetricDataCall::IDX_TRY_GET_DATA
 */
const unsigned int megamol::core::misc::VolumetricDataCall::IDX_TRY_GET_DATA
    = 5;


/*
 * megamol::core::misc::VolumetricDataCall::VolumetricDataCall
 */
megamol::core::misc::VolumetricDataCall::VolumetricDataCall(void)
        : cntFrames(0), data(nullptr), metadata(nullptr), vram_volume_name(0) {
}


/*
 * megamol::core::misc::VolumetricDataCall::VolumetricDataCall
 */
megamol::core::misc::VolumetricDataCall::VolumetricDataCall(
        const VolumetricDataCall& rhs) : data(nullptr), metadata(nullptr), vram_volume_name(0) {
    *this = rhs;
}


/*
 * megamol::core::misc::VolumetricDataCall::~VolumetricDataCall
 */
megamol::core::misc::VolumetricDataCall::~VolumetricDataCall(void) {
}


/*
 * megamol::core::misc::VolumetricDataCall::GetComponents
 */
size_t megamol::core::misc::VolumetricDataCall::GetComponents(void) const {
    return (this->metadata != nullptr) ? this->metadata->Components : 0;
}


/*
 * megamol::core::misc::VolumetricDataCall::GetFrames
 */
size_t megamol::core::misc::VolumetricDataCall::GetFrames(void) const {
    return (this->metadata != nullptr) ? this->metadata->NumberOfFrames : 0;
}


/*
 * megamol::core::misc::VolumetricDataCall::GetGridType
 */
megamol::core::misc::VolumetricDataCall::GridType
megamol::core::misc::VolumetricDataCall::GetGridType(void) const {
    return (this->metadata != nullptr)
        ? this->metadata->GridType
        : GridType::NONE;
}


/*
 * megamol::core::misc::VolumetricDataCall::GetResolution
 */
const size_t megamol::core::misc::VolumetricDataCall::GetResolution(
        const int axis) const {
    if ((axis < 0) || (axis > 2)) {
        throw vislib::OutOfRangeException(axis, 0, 2, __FILE__, __LINE__);
    }
    return (this->metadata != nullptr) ? this->metadata->Resolution[axis] : 0;
}


/*
 * megamol::core::misc::VolumetricDataCall::GetScalarLength
 */
size_t megamol::core::misc::VolumetricDataCall::GetScalarLength(void) const {
    return (this->metadata != nullptr) ? this->metadata->ScalarLength : 0;
}


/*
 * megamol::core::misc::VolumetricDataCall::GetScalarType
 */
megamol::core::misc::VolumetricDataCall::ScalarType
megamol::core::misc::VolumetricDataCall::GetScalarType(void) const {
    return (this->metadata != nullptr)
        ? this->metadata->ScalarType
        : ScalarType::UNKNOWN;
}


/*
 * megamol::core::misc::VolumetricDataCall::GetSliceDistances
 */
const float *megamol::core::misc::VolumetricDataCall::GetSliceDistances(
        const int axis) const {
    if ((axis < 0) || (axis > 2)) {
        throw vislib::OutOfRangeException(axis, 0, 2, __FILE__, __LINE__);
    }
    return (this->metadata != nullptr)
        ? this->metadata->SliceDists[axis]
        : nullptr;
}


/*
 * megamol::core::misc::VolumetricDataCall::GetVoxelsPerFrame
 */
size_t megamol::core::misc::VolumetricDataCall::GetVoxelsPerFrame(void) const {
    if (this->metadata == nullptr) {
        return 0;
    } else {
        size_t retval = 1;
        for (int i = 0; i < STATIC_ARRAY_COUNT(this->metadata->Resolution);
                ++i) {
            retval *= this->metadata->Resolution[i];
        }
        return retval;
    }
}


/*
 * megamol::core::misc::VolumetricDataCall::IsUniform
 */
bool megamol::core::misc::VolumetricDataCall::IsUniform(const int axis) const {
    if ((axis < 0) || (axis > 2)) {
        throw vislib::OutOfRangeException(axis, 0, 2, __FILE__, __LINE__);
    }
    return (this->metadata != nullptr) ? this->metadata->IsUniform[axis] : true;
}


/*
 * megamol::core::misc::VolumetricDataCall::SetMetadata
 */
void megamol::core::misc::VolumetricDataCall::SetMetadata(
        const Metadata *metadata) {
    this->metadata = metadata;
}


/*
 * megamol::core::misc::VolumetricDataCall::operator =
 */
megamol::core::misc::VolumetricDataCall&
megamol::core::misc::VolumetricDataCall::operator =(const VolumetricDataCall& rhs) {
    if (this != &rhs) {
        Base::operator =(rhs);
        this->cntFrames = rhs.cntFrames;
        this->data = rhs.data;
        this->metadata = rhs.metadata;
    }
    return *this;
}


/*
 * megamol::core::misc::VolumetricDataCall::FUNCTIONS
 */
const char *megamol::core::misc::VolumetricDataCall::FUNCTIONS[] = {
    "GetExtents",
    "GetData",
    "GetMetadata",
    "StartAsync",
    "StopAsync",
    "TryGetData"
};
