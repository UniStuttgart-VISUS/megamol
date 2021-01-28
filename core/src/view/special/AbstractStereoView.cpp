/*
 * AbstractStereoView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/special/AbstractStereoView.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"

using namespace megamol::core;


/*
 * view::special::AbstractStereoView::AbstractStereoView
 */
view::special::AbstractStereoView::AbstractStereoView(void) : AbstractOverrideView(),
        projTypeSlot("projType", "The stereo projection type"),
        switchEyesSlot("switchEyes", "Flag to switch the images for the right eye and the left eye") {

    param::EnumParam *ep = new param::EnumParam(
        static_cast<int>(thecam::Projection_type::off_axis));
    ep->SetTypePair(
        static_cast<int>(thecam::Projection_type::off_axis),
        "Off Axis");
    ep->SetTypePair(static_cast<int>(thecam::Projection_type::parallel),
        "Parallel");
    ep->SetTypePair(static_cast<int>(thecam::Projection_type::toe_in),
        "Toe In");
    this->projTypeSlot << ep;
    this->MakeSlotAvailable(&this->projTypeSlot);

    this->switchEyesSlot << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->switchEyesSlot);
}


/*
 * view::special::AbstractStereoView::~AbstractStereoView
 */
view::special::AbstractStereoView::~AbstractStereoView(void) {
    // TODO: Implement
}


/*
 * view::special::AbstractStereoView::packMouseCoordinates
 */
void view::special::AbstractStereoView::packMouseCoordinates(float &x, float &y) {
    x /= this->getViewportWidth();
    y /= this->getViewportHeight();
}


/*
 * view::special::AbstractStereoView::getProjectionType
 */
thecam::Projection_type view::special::AbstractStereoView::getProjectionType(void) const {
    return static_cast<thecam::Projection_type>(
        this->projTypeSlot.Param<param::EnumParam>()->Value());
}


/*
 * view::special::AbstractStereoView::getSwitchEyes
 */
bool view::special::AbstractStereoView::getSwitchEyes(void) const {
    return this->switchEyesSlot.Param<param::BoolParam>()->Value();
}
