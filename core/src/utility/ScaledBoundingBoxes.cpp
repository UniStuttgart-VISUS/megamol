/*
 * ScaledBoundingBoxes.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/utility/ScaledBoundingBoxes.h"

#include "mmcore/BoundingBoxes.h"

#include "glad/glad.h"

#include <stdexcept>
#include <vector>

namespace megamol {
namespace core {
namespace utility {

namespace {
float magicScale(const BoundingBoxes& bb) {
    float scale = bb.ObjectSpaceBBox().LongestEdge();

    if (scale > 0.0000001) {
        //scale = 10.0f / scale;
        scale = 1.0f / scale;
    } else {
        scale = 1.0f;
    }

    return scale;
}
}

BoundingBoxes scaleBoundingBoxes(float scale, const BoundingBoxes& bb) {
    auto out = bb;
    out.MakeScaledWorld(scale);

    return out;
}

BoundingBoxes combineAndScaleBoundingBoxes(float scale, const std::vector<BoundingBoxes>& bbs) {
    if (bbs.empty()) throw std::runtime_error("List of bounding boxes must not be empty");

    auto out = bbs.front();

    for (auto& bb : bbs) {
        auto osbb = out.ObjectSpaceBBox();
        auto oscb = out.ObjectSpaceClipBox();

        osbb.Union(bb.ObjectSpaceBBox());
        oscb.Union(bb.ObjectSpaceClipBox());

        out.SetObjectSpaceBBox(osbb);
        out.SetObjectSpaceClipBox(oscb);
    }

    out.MakeScaledWorld(scale);

    return out;
}

BoundingBoxes magicScaleBoundingBoxes(const BoundingBoxes& bb) { return scaleBoundingBoxes(magicScale(bb), bb); }

BoundingBoxes combineAndMagicScaleBoundingBoxes(const std::vector<BoundingBoxes>& bbs) {
    if (bbs.empty()) throw std::runtime_error("List of bounding boxes must not be empty");

    auto out = bbs.front();

    for (auto& bb : bbs) {
        auto osbb = out.ObjectSpaceBBox();
        auto oscb = out.ObjectSpaceClipBox();

        osbb.Union(bb.ObjectSpaceBBox());
        oscb.Union(bb.ObjectSpaceClipBox());

        out.SetObjectSpaceBBox(osbb);
        out.SetObjectSpaceClipBox(oscb);
    }

    out.MakeScaledWorld(magicScale(out));

    return out;
}

glMagicScale::glMagicScale() : scale(1.0f) { }

glMagicScale::~glMagicScale() {
    glScalef(1.0f / this->scale, 1.0f / this->scale, 1.0f / this->scale);
}

void glMagicScale::apply(const BoundingBoxes& bb) {
    glScalef(1.0f / this->scale, 1.0f / this->scale, 1.0f / this->scale);

    this->scale = magicScale(bb);

    glScalef(this->scale, this->scale, this->scale);
};

}
}
}