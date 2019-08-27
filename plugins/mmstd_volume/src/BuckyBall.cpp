/*
 * BuckyBall.cpp
 *
 * Copyright (C) 2011-2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "BuckyBall.h"

#include "mmcore/Call.h"
#include "mmcore/misc/VolumetricDataCall.h"

#include <limits>

/*
 * megamol::stdplugin::volume::BuckyBall::BuckyBall
 */
megamol::stdplugin::volume::BuckyBall::BuckyBall(void)
	: getDataSlot("getData", "Gets the data")
	, resolution({ 64, 64, 64 })
	, sliceDists({ 4.0f / (resolution[0] - 1), 4.0f / (resolution[1] - 1), 4.0f / (resolution[2] - 1) })
	, minValue(0.0f)
	, maxValue(1.0f) {

    this->getDataSlot.SetCallback(core::misc::VolumetricDataCall::ClassName(),
		core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_DATA), &BuckyBall::getDataCallback);
    this->getDataSlot.SetCallback(core::misc::VolumetricDataCall::ClassName(),
		core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_EXTENTS), &BuckyBall::getExtentCallback);
	this->getDataSlot.SetCallback(core::misc::VolumetricDataCall::ClassName(),
		core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_GET_METADATA), &BuckyBall::getDataCallback);
	this->getDataSlot.SetCallback(core::misc::VolumetricDataCall::ClassName(),
		core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_START_ASYNC), &BuckyBall::getDummyCallback);
	this->getDataSlot.SetCallback(core::misc::VolumetricDataCall::ClassName(),
		core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_STOP_ASYNC), &BuckyBall::getDummyCallback);
	this->getDataSlot.SetCallback(core::misc::VolumetricDataCall::ClassName(),
		core::misc::VolumetricDataCall::FunctionName(core::misc::VolumetricDataCall::IDX_TRY_GET_DATA), &BuckyBall::getDummyCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

	this->volume.resize(this->resolution[0] * this->resolution[1] * this->resolution[2]);
}


/*
 * megamol::stdplugin::volume::BuckyBall::~BuckyBall
 */
megamol::stdplugin::volume::BuckyBall::~BuckyBall(void) {
    this->Release();
}


/*
 * megamol::stdplugin::volume::BuckyBall::create
 */
bool megamol::stdplugin::volume::BuckyBall::create(void) {

    // Generate distance field for truncated icosahedron
    // The sixty vertices of the truncated icosahedron inside a sphere with radius 1
    const unsigned int vertCnt = 60;
    const float vert[] = {
         0.00000f,  0.17952f,  0.98375f,
         0.00000f, -0.17952f,  0.98375f,
         0.34664f, -0.39376f,  0.85135f,
         0.63711f, -0.21424f,  0.74040f,
         0.63711f,  0.21424f,  0.74040f,
         0.34664f,  0.39376f,  0.85135f,
         0.85135f,  0.34664f,  0.39376f,
         0.74040f,  0.63711f,  0.21424f,
         0.39376f,  0.85135f,  0.34664f,
         0.21424f,  0.74040f,  0.63711f,
         0.17952f,  0.98375f, -0.00000f,
        -0.17952f,  0.98375f, -0.00000f,
        -0.39376f,  0.85135f,  0.34664f,
        -0.21424f,  0.74040f,  0.63711f,
        -0.74040f,  0.63711f,  0.21424f,
        -0.85135f,  0.34664f,  0.39376f,
        -0.63711f,  0.21424f,  0.74040f,
        -0.34664f,  0.39376f,  0.85135f,
        -0.63711f, -0.21424f,  0.74040f,
        -0.34664f, -0.39376f,  0.85135f,
        -0.85135f, -0.34664f,  0.39376f,
        -0.74040f, -0.63711f,  0.21424f,
        -0.39376f, -0.85135f,  0.34664f,
        -0.21424f, -0.74040f,  0.63711f,
        -0.17952f, -0.98375f,  0.00000f,
         0.17952f, -0.98375f,  0.00000f,
         0.39376f, -0.85135f,  0.34664f,
         0.21424f, -0.74040f,  0.63711f,
         0.74040f, -0.63711f,  0.21424f,
         0.85135f, -0.34664f,  0.39376f,
         0.98375f, -0.00000f,  0.17952f,
         0.98375f, -0.00000f, -0.17952f,
         0.85135f,  0.34664f, -0.39376f,
         0.74040f,  0.63711f, -0.21424f,
         0.74040f, -0.63711f, -0.21424f,
         0.85135f, -0.34664f, -0.39376f,
        -0.39376f, -0.85135f, -0.34664f,
        -0.21424f, -0.74040f, -0.63711f,
         0.21424f, -0.74040f, -0.63711f,
         0.39376f, -0.85135f, -0.34664f,
         0.34664f, -0.39376f, -0.85135f,
         0.63711f, -0.21424f, -0.74040f,
         0.63711f,  0.21424f, -0.74040f,
         0.34664f,  0.39376f, -0.85135f,
         0.21424f,  0.74040f, -0.63711f,
         0.39376f,  0.85135f, -0.34664f,
         0.00000f, -0.17952f, -0.98375f,
         0.00000f,  0.17952f, -0.98375f,
        -0.74040f, -0.63711f, -0.21424f,
        -0.85135f, -0.34664f, -0.39376f,
        -0.63711f, -0.21424f, -0.74040f,
        -0.34664f, -0.39376f, -0.85135f,
        -0.63711f,  0.21424f, -0.74040f,
        -0.34664f,  0.39376f, -0.85135f,
        -0.21424f,  0.74040f, -0.63711f,
        -0.39376f,  0.85135f, -0.34664f,
        -0.85135f,  0.34664f, -0.39376f,
        -0.74040f,  0.63711f, -0.21424f,
        -0.98375f,  0.00000f,  0.17952f,
        -0.98375f,  0.00000f, -0.17952f
    };

    for (unsigned int x = 0; x < this->resolution[0]; x++) {
        float fx = (static_cast<float>(x) / static_cast<float>(this->resolution[0] - 1)) * 4.0f - 2.0f;

        for (unsigned int y = 0; y < this->resolution[1]; y++) {
            float fy = (static_cast<float>(y) / static_cast<float>(this->resolution[1] - 1)) * 4.0f - 2.0f;

            for (unsigned int z = 0; z < this->resolution[2]; z++) {
                float fz = (static_cast<float>(z) / static_cast<float>(this->resolution[2] - 1)) * 4.0f - 2.0f;

				float value = std::numeric_limits<float>::max();

                for (unsigned int v = 0; v < vertCnt; v++) {
                    float dx = fx - vert[v * 3 + 0];
                    float dy = fy - vert[v * 3 + 1];
                    float dz = fz - vert[v * 3 + 2];

                    float dist = sqrt(dx * dx + dy * dy + dz * dz);
                    if (dist < value) value = dist;
                }

                this->volume[x + this->resolution[0] * (y + this->resolution[1] * z)] = value;
            }
        }
    }

    return true;
}


/*
 * megamol::stdplugin::volume::BuckyBall::release
 */
void megamol::stdplugin::volume::BuckyBall::release(void) {
}


/*
 * megamol::stdplugin::volume::BuckyBall::getDataCallback
 */
bool megamol::stdplugin::volume::BuckyBall::getDataCallback(core::Call& caller) {
    auto *cvd = dynamic_cast<core::misc::VolumetricDataCall*>(&caller);
    if (cvd == nullptr) return false;

	this->metaData.GridType = core::misc::GridType_t::CARTESIAN;
	this->metaData.Resolution[0] = this->resolution[0];
	this->metaData.Resolution[1] = this->resolution[1];
	this->metaData.Resolution[2] = this->resolution[2];
	this->metaData.ScalarType = core::misc::ScalarType_t::FLOATING_POINT;
	this->metaData.ScalarLength = sizeof(float);
	this->metaData.Components = 1;
	this->metaData.SliceDists[0] = const_cast<float*>(&this->sliceDists[0]);
	this->metaData.SliceDists[1] = const_cast<float*>(&this->sliceDists[1]);
	this->metaData.SliceDists[2] = const_cast<float*>(&this->sliceDists[2]);
	this->metaData.Origin[0] = -2.0f;
	this->metaData.Origin[1] = -2.0f;
	this->metaData.Origin[2] = -2.0f;
	this->metaData.IsUniform[0] = true;
	this->metaData.IsUniform[1] = true;
	this->metaData.IsUniform[2] = true;
	this->metaData.NumberOfFrames = 1;
	this->metaData.Extents[0] = 4.0f;
	this->metaData.Extents[1] = 4.0f;
	this->metaData.Extents[2] = 4.0f;
	this->metaData.MinValues = const_cast<double*>(&this->minValue);
	this->metaData.MaxValues = const_cast<double*>(&this->maxValue);
	this->metaData.MemLoc = core::misc::MemoryLocation::RAM;

	cvd->SetMetadata(&this->metaData);
	cvd->SetData(this->volume.data());

    return true;
}


/*
 * megamol::core::BuckyBall::getExtentCallback
 */
bool megamol::stdplugin::volume::BuckyBall::getExtentCallback(core::Call& caller) {
	auto *cvd = dynamic_cast<core::misc::VolumetricDataCall*>(&caller);
    if (cvd == nullptr) return false;

    cvd->AccessBoundingBoxes().Clear();
	cvd->AccessBoundingBoxes().SetObjectSpaceClipBox(-2.0f, -2.0f, -2.0f, 2.0f, 2.0f, 2.0f);
    cvd->AccessBoundingBoxes().SetObjectSpaceBBox(-2.0f, -2.0f, -2.0f, 2.0f, 2.0f, 2.0f);
    cvd->SetDataHash(1);
    cvd->SetFrameCount(1);
	cvd->SetFrameID(0);

    return true;
}


/*
* megamol::core::BuckyBall::getDummyCallback
*/
bool megamol::stdplugin::volume::BuckyBall::getDummyCallback(core::Call& caller) {
	return false;
}
