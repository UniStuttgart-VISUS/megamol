/*
 * DirPartVolume.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define USE_MATH_DEFINES
#include "DirPartVolume.h"
#include "CallVolumeData.h"
#include "moldyn/DirectionalParticleDataCall.h"
#include "param/IntParam.h"
#include "param/FloatParam.h"
#include <cmath>

using namespace megamol::core;


/*
 * moldyn::DirPartVolume::DirPartVolume
 */
moldyn::DirPartVolume::DirPartVolume(void) : Module(),
        inDataSlot("inData", "Connects to the data source"),
        outDataSlot("outData", "Connects this data source"),
        xResSlot("resX", "Number of sample points in x direction"),
        yResSlot("resY", "Number of sample points in y direction"),
        zResSlot("resZ", "Number of sample points in z direction"),
        sampleRadiusSlot("radius", "Radius of the influence range of each particle in object space"),
        dataHash(0), frameID(0), bbox(), data(NULL) {

    this->inDataSlot.SetCompatibleCall<moldyn::DirectionalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    this->outDataSlot.SetCallback("CallVolumeData", "GetData", &DirPartVolume::outData);
    this->outDataSlot.SetCallback("CallVolumeData", "GetExtent", &DirPartVolume::outExtend);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->xResSlot << new param::IntParam(64, 2);
    this->MakeSlotAvailable(&this->xResSlot);

    this->yResSlot << new param::IntParam(64, 2);
    this->MakeSlotAvailable(&this->yResSlot);

    this->zResSlot << new param::IntParam(64, 2);
    this->MakeSlotAvailable(&this->zResSlot);

    this->sampleRadiusSlot << new param::FloatParam(2.0f, 0.0f);
    this->MakeSlotAvailable(&this->sampleRadiusSlot);

}


/*
 * moldyn::DirPartVolume::~DirPartVolume
 */
moldyn::DirPartVolume::~DirPartVolume(void) {
    this->Release();
}


/*
 * moldyn::DirPartVolume::create
 */
bool moldyn::DirPartVolume::create(void) {
    // intentionally empty
    return true;
}


/*
 * moldyn::DirPartVolume::release
 */
void moldyn::DirPartVolume::release(void) {
    ARY_SAFE_DELETE(this->data);
}


/*
 * moldyn::DirPartVolume::outExtend
 */
bool moldyn::DirPartVolume::outExtend(Call& caller) {
    CallVolumeData *cvd = dynamic_cast<CallVolumeData*>(&caller);
    if (cvd == NULL) return false;

    cvd->AccessBoundingBoxes().Clear();
    DirectionalParticleDataCall *dpd = this->inDataSlot.CallAs<DirectionalParticleDataCall>();
    if ((dpd == NULL) || (!(*dpd)(1))) {
        // no input data
        cvd->SetDataHash(0);
        cvd->SetFrameCount(1);

    } else {
        // input data in dpd
        cvd->SetDataHash(dpd->DataHash());
        cvd->SetFrameCount(dpd->FrameCount());
        this->bbox = dpd->AccessBoundingBoxes().ObjectSpaceBBox();

        float sx = 0.5f / static_cast<float>(this->xResSlot.Param<param::IntParam>()->Value());
        float sy = 0.5f / static_cast<float>(this->yResSlot.Param<param::IntParam>()->Value());
        float sz = 0.5f / static_cast<float>(this->zResSlot.Param<param::IntParam>()->Value());

        // voxel at cell center positions, I say ...
        this->bbox.Grow(
            -sx * this->bbox.Width(),
            -sy * this->bbox.Height(),
            -sz * this->bbox.Depth());

        cvd->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    }
    cvd->SetUnlocker(NULL);

    return true;
}


/*
 * moldyn::DirPartVolume::outData
 */
bool moldyn::DirPartVolume::outData(Call& caller) {
    CallVolumeData *cvd = dynamic_cast<CallVolumeData*>(&caller);
    if (cvd == NULL) return false;

    DirectionalParticleDataCall *dpd = this->inDataSlot.CallAs<DirectionalParticleDataCall>();
    if (dpd == NULL) return false;
    dpd->SetFrameID(cvd->FrameID(), cvd->IsFrameForced());
    if (!(*dpd)(0)) return false;

    // We have data!

    bool rebuild = false;
    if (this->xResSlot.IsDirty()) {
        this->xResSlot.ResetDirty();
        rebuild = true;
    }
    if (this->yResSlot.IsDirty()) {
        this->yResSlot.ResetDirty();
        rebuild = true;
    }
    if (this->zResSlot.IsDirty()) {
        this->zResSlot.ResetDirty();
        rebuild = true;
    }
    unsigned int sx = static_cast<unsigned int>(this->xResSlot.Param<param::IntParam>()->Value());
    unsigned int sy = static_cast<unsigned int>(this->yResSlot.Param<param::IntParam>()->Value());
    unsigned int sz = static_cast<unsigned int>(this->zResSlot.Param<param::IntParam>()->Value());

    if (this->dataHash != dpd->DataHash()) {
        rebuild = true;
        this->dataHash = dpd->DataHash();
    }
    if (this->frameID != dpd->FrameID()) {
        rebuild = true;
        this->frameID = dpd->FrameID();
    }
    if (this->sampleRadiusSlot.IsDirty()) {
        this->sampleRadiusSlot.ResetDirty();
        rebuild = true;
    }

    const unsigned int attrCnt = 2;
    //  0 : merged vector length
    //  1 : symmetric merged vector length

    if (rebuild) {
        delete[] this->data;
        this->data = new float[sx * sy * sz * attrCnt];

        float rad = this->sampleRadiusSlot.Param<param::FloatParam>()->Value();

        // TODO: Implement

    }

    cvd->SetAttributeCount(attrCnt);
    cvd->Attribute(0).SetName("o");
    cvd->Attribute(0).SetType(CallVolumeData::TYPE_FLOAT);
    cvd->Attribute(0).SetData(&this->data[sx * sy * sz * 0]);
    cvd->Attribute(1).SetName("s");
    cvd->Attribute(1).SetType(CallVolumeData::TYPE_FLOAT);
    cvd->Attribute(1).SetData(&this->data[sx * sy * sz * 1]);
    cvd->SetDataHash(this->dataHash);
    cvd->SetFrameID(this->frameID);
    cvd->SetSize(sx, sy, sz);
    cvd->SetUnlocker(NULL);

    return true;
}
