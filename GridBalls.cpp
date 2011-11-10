/*
 * GridBalls.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GridBalls.h"
#include "CallVolumeData.h"
#include "moldyn/MultiParticleDataCall.h"
#include "param/StringParam.h"
#include "param/FloatParam.h"
#include <climits>
#include <cfloat>
#include <cmath>


/*
 * megamol::core::GridBalls::GridBalls
 */
megamol::core::GridBalls::GridBalls(void) : 
        inDataSlot("inData", "The slot for requesting input data"),
        outDataSlot("outData", "Gets the data"), 
        attributeSlot("attr", "The attribute to show"),
        radiusSlot("rad", "The spheres' radius"),
        lowValSlot("low", "The low value"),
        highValSlot("high", "The high value"),
        grid(NULL), sx(0), sy(0), sz(0), osbb() {

    this->inDataSlot.SetCompatibleCall<CallVolumeDataDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    this->outDataSlot.SetCallback("MultiParticleDataCall", "GetData", &GridBalls::outDataCallback);
    this->outDataSlot.SetCallback("MultiParticleDataCall", "GetExtent", &GridBalls::outExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->attributeSlot << new param::StringParam("0");
    this->MakeSlotAvailable(&this->attributeSlot);

    this->radiusSlot << new param::FloatParam(0.1f, 0.0f);
    this->MakeSlotAvailable(&this->radiusSlot);

    this->lowValSlot << new param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->lowValSlot);

    this->highValSlot << new param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->highValSlot);

}


/*
 * megamol::core::GridBalls::~GridBalls
 */
megamol::core::GridBalls::~GridBalls(void) {
    this->Release();
}


/*
 * megamol::core::GridBalls::create
 */
bool megamol::core::GridBalls::create(void) {
    // intentionally empty
    return true;
}


/*
 * megamol::core::GridBalls::release
 */
void megamol::core::GridBalls::release(void) {
    ARY_SAFE_DELETE(this->grid);
}


/*
 * megamol::core::GridBalls::outDataCallback
 */
bool megamol::core::GridBalls::outDataCallback(megamol::core::Call& caller) {
    moldyn::MultiParticleDataCall *mpdc = dynamic_cast<moldyn::MultiParticleDataCall*>(&caller);
    if (mpdc == NULL) return false;

    mpdc->AccessBoundingBoxes().Clear();
    CallVolumeData *cvd = this->inDataSlot.CallAs<CallVolumeData>();
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
        vislib::StringA attrName(this->attributeSlot.Param<param::StringParam>()->Value());
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

        float r = this->radiusSlot.Param<param::FloatParam>()->Value();

        moldyn::SimpleSphericalParticles& parts = mpdc->AccessParticles(0);
        parts.SetCount(this->sx * this->sy * this->sz);
        if (this->grid != NULL) {
            parts.SetVertexData(moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ, this->grid);
        } else {
            parts.SetVertexData(moldyn::SimpleSphericalParticles::VERTDATA_NONE, NULL);
        }
        if (attrIdx == UINT_MAX) {
            parts.SetColourData(moldyn::SimpleSphericalParticles::COLDATA_NONE, NULL);
            parts.SetGlobalColour(192, 192, 192);
        } else {
            parts.SetColourData(moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I, cvd->Attribute(attrIdx).Floats());
            parts.SetColourMapIndexValues(
                this->lowValSlot.Param<param::FloatParam>()->Value(),
                this->highValSlot.Param<param::FloatParam>()->Value());
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
 * megamol::core::GridBalls::outExtentCallback
 */
bool megamol::core::GridBalls::outExtentCallback(megamol::core::Call& caller) {
    moldyn::MultiParticleDataCall *mpdc = dynamic_cast<moldyn::MultiParticleDataCall*>(&caller);
    if (mpdc == NULL) return false;

    mpdc->AccessBoundingBoxes().Clear();
    CallVolumeData *cvd = this->inDataSlot.CallAs<CallVolumeData>();
    if ((cvd == NULL) || (!(*cvd)(1))) {
        // no input data
        mpdc->SetDataHash(0);
        mpdc->SetFrameCount(1);

    } else {
        // input data in cvd
        mpdc->SetDataHash(cvd->DataHash());
        mpdc->SetExtent(cvd->FrameCount(), cvd->AccessBoundingBoxes());
        float r = this->radiusSlot.Param<param::FloatParam>()->Value();
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
