/*
 * BezierControlLines.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "misc/BezierControlLines.h"
#include "vislib/NamedColours.h"

using namespace megamol::core;


/*
 * misc::BezierControlLines::BezierControlLines
 */
misc::BezierControlLines::BezierControlLines(void) : Module(),
        dataSlot("data", "Provides with line data"), vertData(), idxData(),
        hash(0), lines() {

    this->dataSlot.SetCallback(LinesDataCall::ClassName(), "GetData",
        &BezierControlLines::getDataCallback);
    this->dataSlot.SetCallback(LinesDataCall::ClassName(), "GetExtent",
        &BezierControlLines::getExtentCallback);
    this->MakeSlotAvailable(&this->dataSlot);

}


/*
 * misc::BezierControlLines::~BezierControlLines
 */
misc::BezierControlLines::~BezierControlLines(void) {
    this->Release();
}


/*
 * misc::BezierControlLines::create
 */
bool misc::BezierControlLines::create(void) {

    // TODO: Implement

    return true;
}


/*
 * misc::BezierControlLines::release
 */
void misc::BezierControlLines::release(void) {

    // TODO: Implement

}


/*
 * misc::BezierControlLines::getDataCallback
 */
bool misc::BezierControlLines::getDataCallback(Call& call) {
    LinesDataCall *ldc = dynamic_cast<LinesDataCall*>(&call);
    if (ldc == NULL) return false;

    SIZE_T inHash = 0; // TODO: Implement

    if ((inHash == 0) || (inHash != this->hash)) {
        this->hash = inHash;
        this->vertData.EnforceSize(0);
        this->idxData.EnforceSize(0);

        // TODO: Implement

        this->vertData.EnforceSize(4 * 3 * sizeof(float));

        *this->vertData.AsAt<float>(0 + 0) = -0.5f;
        *this->vertData.AsAt<float>(0 + 4) = 0.0f;
        *this->vertData.AsAt<float>(0 + 8) = -0.5f;

        *this->vertData.AsAt<float>(12 + 0) = 0.5f;
        *this->vertData.AsAt<float>(12 + 4) = 0.0f;
        *this->vertData.AsAt<float>(12 + 8) = -0.5f;

        *this->vertData.AsAt<float>(24 + 0) = 0.0f;
        *this->vertData.AsAt<float>(24 + 4) = -0.5f;
        *this->vertData.AsAt<float>(24 + 8) = 0.5f;

        *this->vertData.AsAt<float>(36 + 0) = 0.0f;
        *this->vertData.AsAt<float>(36 + 4) = 0.5f;
        *this->vertData.AsAt<float>(36 + 8) = 0.5f;

        this->idxData.EnforceSize(6 * 2 * sizeof(unsigned int));

        *this->idxData.AsAt<unsigned int>(0 + 0) = 0;
        *this->idxData.AsAt<unsigned int>(0 + 4) = 1;

        *this->idxData.AsAt<unsigned int>(8 + 0) = 0;
        *this->idxData.AsAt<unsigned int>(8 + 4) = 2;

        *this->idxData.AsAt<unsigned int>(16 + 0) = 0;
        *this->idxData.AsAt<unsigned int>(16 + 4) = 3;

        *this->idxData.AsAt<unsigned int>(24 + 0) = 1;
        *this->idxData.AsAt<unsigned int>(24 + 4) = 2;

        *this->idxData.AsAt<unsigned int>(32 + 0) = 1;
        *this->idxData.AsAt<unsigned int>(32 + 4) = 3;

        *this->idxData.AsAt<unsigned int>(40 + 0) = 2;
        *this->idxData.AsAt<unsigned int>(40 + 4) = 3;

        this->lines.Set(
            static_cast<unsigned int>(this->idxData.GetSize() / sizeof(unsigned int)),
            this->idxData.As<unsigned int>(), this->vertData.As<float>(),
            vislib::graphics::NamedColours::AliceBlue);
    }

    ldc->SetData(1, &this->lines);
    ldc->SetDataHash(inHash);

    return true;
}


/*
 * misc::BezierControlLines::getExtentCallback
 */
bool misc::BezierControlLines::getExtentCallback(Call& call) {
    LinesDataCall *ldc = dynamic_cast<LinesDataCall*>(&call);
    if (ldc == NULL) return false;

    ldc->AccessBoundingBoxes().Clear();
    ldc->SetFrameCount(1);
    ldc->SetDataHash(0);

    // TODO: Implement

    return true;
}
