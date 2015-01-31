/*
 * GridBalls.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GridBalls.h"
#include "mmcore/CallVolumeData.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/FloatParam.h"
#include <climits>
#include <cfloat>
#include <cmath>


/*
 * megamol::stdplugin::volume::GridBalls::GridBalls
 */
megamol::stdplugin::volume::GridBalls::GridBalls(void) : 
        inDataSlot("inData", "The slot for requesting input data"),
        outDataSlot("outData", "Gets the data"), 
        attributeSlot("attr", "The attribute to show"),
        radiusSlot("rad", "The spheres' radius"),
        lowValSlot("low", "The low value"),
        highValSlot("high", "The high value"),
        grid(NULL), sx(0), sy(0), sz(0), osbb() {

    this->inDataSlot.SetCompatibleCall<core::CallVolumeDataDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    this->outDataSlot.SetCallback("MultiParticleDataCall", "GetData", &GridBalls::outDataCallback);
    this->outDataSlot.SetCallback("MultiParticleDataCall", "GetExtent", &GridBalls::outExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->attributeSlot << new core::param::StringParam("0");
    this->MakeSlotAvailable(&this->attributeSlot);

    this->radiusSlot << new core::param::FloatParam(0.1f, 0.0f);
    this->MakeSlotAvailable(&this->radiusSlot);

    this->lowValSlot << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->lowValSlot);

    this->highValSlot << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->highValSlot);

}


/*
 * megamol::stdplugin::volume::GridBalls::~GridBalls
 */
megamol::stdplugin::volume::GridBalls::~GridBalls(void) {
    this->Release();
}


/*
 * megamol::stdplugin::volume::GridBalls::create
 */
bool megamol::stdplugin::volume::GridBalls::create(void) {
    // intentionally empty
    return true;
}


/*
 * megamol::stdplugin::volume::GridBalls::release
 */
void megamol::stdplugin::volume::GridBalls::release(void) {
    ARY_SAFE_DELETE(this->grid);
}


/*
 * megamol::stdplugin::volume::GridBalls::outDataCallback
 */
bool megamol::stdplugin::volume::GridBalls::outDataCallback(megamol::core::Call& caller) {
    core::moldyn::MultiParticleDataCall *mpdc = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&caller);
    if (mpdc == NULL) return false;

    mpdc->AccessBoundingBoxes().Clear();
    core::CallVolumeData *cvd = this->inDataSlot.CallAs<core::CallVolumeData>();
    if (cvd != NULL) {
        cvd->SetFrameID(mpdc->FrameID(), mpdc->IsFrameForced());
        if ((*cvd)(1)) {
            if (this->osbb != cvd->AccessBoundingBoxes().ObjectSpaceBBox()) {
                this->osbb = cvd->AccessBoundingBoxes().ObjectSpaceBBox();
                this->sx = 0;
            }
            cvd->SetFrameID(mpdc->FrameID(), mpdc->IsFrameForced());
        }
        if (!(*cvd)(0)) {
            cvd = NULL; // DO NOT DELETE
        }
    }

    unsigned int attrIdx = UINT_MAX;
    if (cvd != NULL) {
        if ((cvd->XSize() != this->sx) || (cvd->YSize() != this->sy) || (cvd->ZSize() != this->sz)) {
            ARY_SAFE_DELETE(this->grid);
            this->sx = cvd->XSize();
            this->sy = cvd->YSize();
            this->sz = cvd->ZSize();

            if (this->sx * this->sy * this->sz > 0) {
                // rebuild grid
                this->grid = new float[this->sx * this->sy * this->sz * 3];
                unsigned int off = 0;
                for (unsigned int z = 0; z < this->sz; z++) {
                    float zPos = static_cast<float>(z) / static_cast<float>(this->sz - 1);
                    zPos = (zPos * this->osbb.Depth()) + this->osbb.Back();
                    for (unsigned int y = 0; y < this->sy; y++) {
                        float yPos = static_cast<float>(y) / static_cast<float>(this->sy - 1);
                        yPos = (yPos * this->osbb.Height()) + this->osbb.Bottom();
                        for (unsigned int x = 0; x < this->sx; x++, off += 3) {
                            float xPos = static_cast<float>(x) / static_cast<float>(this->sx - 1);
                            xPos = (xPos * this->osbb.Width()) + this->osbb.Left();

                            this->grid[off + 0] = xPos;
                            this->grid[off + 1] = yPos;
                            this->grid[off + 2] = zPos;

                        }
                    }
                }
            }

        }

        // find attribute
        vislib::StringA attrName(this->attributeSlot.Param<core::param::StringParam>()->Value());
        attrIdx = cvd->FindAttribute(attrName);
        if (attrIdx == UINT_MAX) {
            try {
                attrIdx = static_cast<unsigned int>(vislib::CharTraitsA::ParseInt(attrName));
            } catch(...) {
                attrIdx = UINT_MAX;
            }
        }
    }

    if (cvd != NULL) {
        // has data
        mpdc->SetDataHash(cvd->DataHash());
        mpdc->SetFrameID(cvd->FrameID());
        mpdc->SetParticleListCount(1);

        float r = this->radiusSlot.Param<core::param::FloatParam>()->Value();

        core::moldyn::SimpleSphericalParticles& parts = mpdc->AccessParticles(0);
        parts.SetCount(this->sx * this->sy * this->sz);
        if (this->grid != NULL) {
            parts.SetVertexData(core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, this->grid);
        } else {
            parts.SetVertexData(core::moldyn::SimpleSphericalParticles::VERTDATA_NONE, NULL);
        }
        if (attrIdx == UINT_MAX) {
            parts.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_NONE, NULL);
            parts.SetGlobalColour(192, 192, 192);
        } else {
            parts.SetColourData(core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, cvd->Attribute(attrIdx).Floats());
            parts.SetColourMapIndexValues(
                this->lowValSlot.Param<core::param::FloatParam>()->Value(),
                this->highValSlot.Param<core::param::FloatParam>()->Value());
        }
        parts.SetGlobalRadius(r);

    } else {
        // has not data
        mpdc->SetDataHash(0);
        mpdc->SetParticleListCount(0);

    }

    mpdc->SetUnlocker(NULL);

    return true;
}


/*
 * megamol::stdplugin::volume::GridBalls::outExtentCallback
 */
bool megamol::stdplugin::volume::GridBalls::outExtentCallback(megamol::core::Call& caller) {
    core::moldyn::MultiParticleDataCall *mpdc = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&caller);
    if (mpdc == NULL) return false;

    mpdc->AccessBoundingBoxes().Clear();
    core::CallVolumeData *cvd = this->inDataSlot.CallAs<core::CallVolumeData>();
    if ((cvd == NULL) || (!(*cvd)(1))) {
        // no input data
        mpdc->SetDataHash(0);
        mpdc->SetFrameCount(1);

    } else {
        // input data in cvd
        mpdc->SetDataHash(cvd->DataHash());
        mpdc->SetExtent(cvd->FrameCount(), cvd->AccessBoundingBoxes());
        float r = this->radiusSlot.Param<core::param::FloatParam>()->Value();
        vislib::math::Cuboid<float> bb = 
            mpdc->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()
            ? mpdc->AccessBoundingBoxes().ObjectSpaceClipBox()
            : mpdc->AccessBoundingBoxes().ObjectSpaceBBox();
        bb.Grow(r);
        mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox(bb);

    }
    mpdc->SetUnlocker(NULL);

    return true;
}
