/*
 * AbstractStereoView.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "view/special/AbstractStereoView.h"
#include "param/BoolParam.h"
#include "param/EnumParam.h"

using namespace megamol::core;


/*
 * view::special::AbstractStereoView::AbstractStereoView
 */
view::special::AbstractStereoView::AbstractStereoView(void) : AbstractOverrideView(),
        projTypeSlot("projType", "The stereo projection type"),
        switchEyesSlot("switchEyes", "Flag to switch the images for the right eye and the left eye") {

    param::EnumParam *ep = new param::EnumParam(
        static_cast<int>(vislib::graphics::CameraParameters::STEREO_OFF_AXIS));
    ep->SetTypePair(
        static_cast<int>(vislib::graphics::CameraParameters::STEREO_OFF_AXIS),
        "Off Axis");
    ep->SetTypePair(
        static_cast<int>(vislib::graphics::CameraParameters::STEREO_PARALLEL),
        "Parallel");
    ep->SetTypePair(
        static_cast<int>(vislib::graphics::CameraParameters::STEREO_TOE_IN),
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
vislib::graphics::CameraParameters::ProjectionType view::special::AbstractStereoView::getProjectionType(void) const {
    return static_cast<vislib::graphics::CameraParameters::ProjectionType>(
        this->projTypeSlot.Param<param::EnumParam>()->Value());
}


/*
 * view::special::AbstractStereoView::getSwitchEyes
 */
bool view::special::AbstractStereoView::getSwitchEyes(void) const {
    return this->switchEyesSlot.Param<param::BoolParam>()->Value();
}
