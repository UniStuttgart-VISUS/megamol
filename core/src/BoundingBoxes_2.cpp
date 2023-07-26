/**
 * MegaMol
 * Copyright (c) 2018, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/BoundingBoxes_2.h"

#include "vislib/math/mathfunctions.h"

using namespace megamol::core;

/*
 * BoundingBoxes_2::BoundingBoxes_2
 */
BoundingBoxes_2::BoundingBoxes_2() : clipBox(), clipBoxValid(false), boundingBox(), boundingBoxValid(false) {
    // intentionally empty
}

/*
 * BoundingBoxes_2::BoundingBoxes_2
 */
BoundingBoxes_2::BoundingBoxes_2(const BoundingBoxes_2& src)
        : clipBox(src.clipBox)
        , clipBoxValid(src.clipBoxValid)
        , boundingBox(src.boundingBox)
        , boundingBoxValid(src.boundingBoxValid) {
    // intentionally empty
}

/*
 * BoundingBoxes_2::~BoundingBoxes_2
 */
BoundingBoxes_2::~BoundingBoxes_2() {
    // intentionally empty
}

/*
 * BoundingBoxes_2::operator==
 */
bool BoundingBoxes_2::operator==(const BoundingBoxes_2& rhs) const {
    return (!this->boundingBoxValid || (this->boundingBox == rhs.boundingBox)) &&
           (this->boundingBoxValid == rhs.boundingBoxValid) &&
           (!this->clipBoxValid || (this->clipBox == rhs.clipBox)) && (this->clipBoxValid == rhs.clipBoxValid);
}

/*
 * BoundingBoxes_2::operator=
 */
BoundingBoxes_2& BoundingBoxes_2::operator=(const BoundingBoxes_2& rhs) {
    this->clipBox = rhs.clipBox;
    this->clipBoxValid = rhs.clipBoxValid;
    this->boundingBox = rhs.boundingBox;
    this->boundingBoxValid = rhs.boundingBoxValid;
    return *this;
}

/*
 * BoundingBoxes_2::operator=
 */
BoundingBoxes_2& BoundingBoxes_2::operator=(const BoundingBoxes& rhs) {
    this->clipBox = rhs.ObjectSpaceClipBox();
    this->clipBoxValid = rhs.IsObjectSpaceClipBoxValid();
    this->boundingBox = rhs.ObjectSpaceBBox();
    this->boundingBoxValid = rhs.IsObjectSpaceBBoxValid();
    return *this;
}

/*
 * BoundingBoxes_2::calcClipBox
 */
void BoundingBoxes_2::calcClipBox() const {
    if (!this->clipBoxValid) {
        if (this->boundingBoxValid) {
            this->clipBox = this->boundingBox;
        } else {
            this->clipBox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        }
    }
    this->clipBoxValid = true;
}
