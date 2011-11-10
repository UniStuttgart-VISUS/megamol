/*
 * BuckyBall.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "BuckyBall.h"
#include "CallVolumeData.h"
#include <climits>
#include <cfloat>
#include <cmath>


/*
 * megamol::core::BuckyBall::BuckyBall
 */
megamol::core::BuckyBall::BuckyBall(void) : getDataSlot("getData", "Gets the data"), volRes(128), vol(NULL) {
    this->getDataSlot.SetCallback("CallVolumeData", "GetData", &BuckyBall::getDataCallback);
    this->getDataSlot.SetCallback("CallVolumeData", "GetExtent", &BuckyBall::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

    this->vol = new float[this->volRes * this->volRes * this->volRes];
}


/*
 * megamol::core::BuckyBall::~BuckyBall
 */
megamol::core::BuckyBall::~BuckyBall(void) {
    this->Release();
}


/*
 * megamol::core::BuckyBall::create
 */
bool megamol::core::BuckyBall::create(void) {

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

    // 32 faces; 6 indices each; 1-based; trailing 0 indicates a pentagon
    //const unsigned int facesCnt = 32;
    //const unsigned int faces[] = {
    //     1u,  2u,  3u,  4u,  5u,  6u,
    //     6u,  5u,  7u,  8u,  9u, 10u,
    //    10u,  9u, 11u, 12u, 13u, 14u,
    //    14u, 13u, 15u, 16u, 17u, 18u,
    //    18u, 17u, 19u, 20u,  2u,  1u,
    //    20u, 19u, 21u, 22u, 23u, 24u,
    //    24u, 23u, 25u, 26u, 27u, 28u,
    //    28u, 27u, 29u, 30u,  4u,  3u,
    //     8u,  7u, 31u, 32u, 33u, 34u,
    //    32u, 31u, 30u, 29u, 35u, 36u,
    //    26u, 25u, 37u, 38u, 39u, 40u,
    //    40u, 39u, 41u, 42u, 36u, 35u,
    //    34u, 33u, 43u, 44u, 45u, 46u,
    //    44u, 43u, 42u, 41u, 47u, 48u,
    //    38u, 37u, 49u, 50u, 51u, 52u,
    //    52u, 51u, 53u, 54u, 48u, 47u,
    //    46u, 45u, 55u, 56u, 12u, 11u,
    //    56u, 55u, 54u, 53u, 57u, 58u,
    //    50u, 49u, 22u, 21u, 59u, 60u,
    //    60u, 59u, 16u, 15u, 58u, 57u,
    //     1u,  6u, 10u, 14u, 18u,  0u,
    //     2u, 20u, 24u, 28u,  3u,  0u,
    //     5u,  4u, 30u, 31u,  7u,  0u,
    //    29u, 27u, 26u, 40u, 35u,  0u,
    //    32u, 36u, 42u, 43u, 33u,  0u,
    //    41u, 39u, 38u, 52u, 47u,  0u,
    //    44u, 48u, 54u, 55u, 45u,  0u,
    //    53u, 51u, 50u, 60u, 57u,  0u,
    //    56u, 58u, 15u, 13u, 12u,  0u,
    //    16u, 59u, 21u, 19u, 17u,  0u,
    //    23u, 22u, 49u, 37u, 25u,  0u,
    //    11u,  9u,  8u, 34u, 46u,  0u
    //};

    float val = 0.0f;

    for (unsigned int x = 0; x < this->volRes; x++) {
        float fx = (static_cast<float>(x) / static_cast<float>(this->volRes - 1)) * 4.0f - 2.0f;

        for (unsigned int y = 0; y < this->volRes; y++) {
            float fy = (static_cast<float>(y) / static_cast<float>(this->volRes - 1)) * 4.0f - 2.0f;

            for (unsigned int z = 0; z < this->volRes; z++) {
                float fz = (static_cast<float>(z) / static_cast<float>(this->volRes - 1)) * 4.0f - 2.0f;

                val = FLT_MAX;
                for (unsigned int v = 0; v < vertCnt; v++) {
                    float dx = fx - vert[v * 3 + 0];
                    float dy = fy - vert[v * 3 + 1];
                    float dz = fz - vert[v * 3 + 2];

                    float dist = sqrt(dx * dx + dy * dy + dz * dz);
                    if (dist < val) val = dist;
                }

                this->vol[x + this->volRes * (y + this->volRes * z)] = val;
            }
        }
    }

    return true;
}


/*
 * megamol::core::BuckyBall::release
 */
void megamol::core::BuckyBall::release(void) {
    ARY_SAFE_DELETE(this->vol);
}


/*
 * megamol::core::BuckyBall::getDataCallback
 */
bool megamol::core::BuckyBall::getDataCallback(megamol::core::Call& caller) {
    CallVolumeData *cvd = dynamic_cast<CallVolumeData*>(&caller);
    if (cvd == NULL) return false;

    cvd->SetAttributeCount(1);
    cvd->SetDataHash(1);
    cvd->SetFrameID(0);
    cvd->SetSize(this->volRes, this->volRes, this->volRes);
    cvd->SetUnlocker(NULL);
    cvd->Attribute(0).SetName("d");
    cvd->Attribute(0).SetType(CallVolumeData::TYPE_FLOAT);
    cvd->Attribute(0).SetData(this->vol);

    return true;
}


/*
 * megamol::core::BuckyBall::getExtentCallback
 */
bool megamol::core::BuckyBall::getExtentCallback(megamol::core::Call& caller) {
    CallVolumeData *cvd = dynamic_cast<CallVolumeData*>(&caller);
    if (cvd == NULL) return false;

    cvd->AccessBoundingBoxes().Clear();
    cvd->AccessBoundingBoxes().SetObjectSpaceBBox(-2.0f, -2.0f, -2.0f, 2.0f, 2.0f, 2.0f);
    cvd->SetDataHash(1);
    cvd->SetFrameCount(1);

    return true;
}
