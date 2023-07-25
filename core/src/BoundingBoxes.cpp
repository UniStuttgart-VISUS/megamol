/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/BoundingBoxes.h"
#include "vislib/math/mathfunctions.h"

using namespace megamol::core;


/*
 * BoundingBoxes::BoundingBoxes
 */
BoundingBoxes::BoundingBoxes()
        : clipBox()
        , clipBoxValid(false)
        , osBBox()
        , osBBoxValid(false)
        , osClipBox()
        , osClipBoxValid(false)
        , osScale(0.0)
        , wsBBox()
        , wsBBoxValid(false)
        , wsClipBox()
        , wsClipBoxValid(false) {
    // intentionally empty
}


/*
 * BoundingBoxes::BoundingBoxes
 */
BoundingBoxes::BoundingBoxes(const BoundingBoxes& src)
        : clipBox(src.clipBox)
        , clipBoxValid(src.clipBoxValid)
        , osBBox(src.osBBox)
        , osBBoxValid(src.osBBoxValid)
        , osClipBox(src.osClipBox)
        , osClipBoxValid(src.osClipBoxValid)
        , osScale(src.osScale)
        , wsBBox(src.wsBBox)
        , wsBBoxValid(src.wsBBoxValid)
        , wsClipBox(src.wsClipBox)
        , wsClipBoxValid(src.wsClipBoxValid) {
    // intentionally empty
}


/*
 * BoundingBoxes::~BoundingBoxes
 */
BoundingBoxes::~BoundingBoxes() {
    // intentionally empty
}


/*
 * BoundingBoxes::MakeScaledWorld
 */
void BoundingBoxes::MakeScaledWorld(float f) {
    this->clipBoxValid = false;
    this->wsBBox = this->osBBox;
    this->wsBBox *= f;
    this->wsBBoxValid = this->osBBoxValid;
    this->wsClipBox = this->osClipBox;
    this->wsClipBox *= f;
    this->wsClipBoxValid = this->osClipBoxValid;
}


/*
 * BoundingBoxes::operator==
 */
bool BoundingBoxes::operator==(const BoundingBoxes& rhs) const {
    // don't have to check clip box, because it's evaluated lazy
    return (!this->osBBoxValid || (this->osBBox == rhs.osBBox)) && (this->osBBoxValid == rhs.osBBoxValid) &&
           (!this->osClipBoxValid || (this->osClipBox == rhs.osClipBox)) &&
           (this->osClipBoxValid == rhs.osClipBoxValid) && vislib::math::IsEqual(this->osScale, rhs.osScale) &&
           (!this->wsBBoxValid || (this->wsBBox == rhs.wsBBox)) && (this->wsBBoxValid == rhs.wsBBoxValid) &&
           (!this->wsClipBoxValid || (this->wsClipBox == rhs.wsClipBox)) &&
           (this->wsClipBoxValid == rhs.wsClipBoxValid);
}


/*
 * BoundingBoxes::operator=
 */
BoundingBoxes& BoundingBoxes::operator=(const BoundingBoxes& rhs) {
    this->clipBox = rhs.clipBox;
    this->clipBoxValid = rhs.clipBoxValid;
    this->osBBox = rhs.osBBox;
    this->osBBoxValid = rhs.osBBoxValid;
    this->osClipBox = rhs.osClipBox;
    this->osClipBoxValid = rhs.osClipBoxValid;
    this->osScale = rhs.osScale;
    this->wsBBox = rhs.wsBBox;
    this->wsBBoxValid = rhs.wsBBoxValid;
    this->wsClipBox = rhs.wsClipBox;
    this->wsClipBoxValid = rhs.wsClipBoxValid;
    return *this;
}


/*
 * BoundingBoxes::calcClipBox
 */
void BoundingBoxes::calcClipBox() const {
    if (this->wsClipBoxValid) {
        this->clipBox = this->wsClipBox;
        if (this->wsBBoxValid) {
            this->clipBox.Union(this->wsBBox);
        }
    } else {
        if (this->wsBBoxValid) {
            this->clipBox = this->wsBBox;
        } else {

            if (this->osClipBoxValid) {
                this->clipBox = this->osClipBox;
                if (this->osBBoxValid) {
                    this->clipBox.Union(this->osBBox);
                }
            } else {
                if (this->osBBoxValid) {
                    this->clipBox = this->osBBox;
                } else {

                    // if everything fails ...
                    this->clipBox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
                }
            }
        }
    }
    this->clipBoxValid = true;
}
