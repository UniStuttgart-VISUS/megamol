/*
 * CameraParameters.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */


#include "vislib/CameraParameters.h"
#include "vislib/mathtypes.h"


/*
 * vislib::graphics::CameraParameters::CameraParameters
 */
vislib::graphics::CameraParameters::CameraParameters(void) {
}


/* 
 * vislib::graphics::CameraParameters::CameraParameters 
 */
vislib::graphics::CameraParameters::CameraParameters(
        const vislib::graphics::CameraParameters& rhs) {
    *this = rhs;
}


/*
 * vislib::graphics::CameraParameters::~CameraParameters
 */
vislib::graphics::CameraParameters::~CameraParameters(void) {
}


/*
 * vislib::graphics::CameraParameters::CalcClipping
 */
void vislib::graphics::CameraParameters::CalcClipping(
        const vislib::math::Cuboid<vislib::graphics::SceneSpaceType>& bbox, 
        vislib::graphics::SceneSpaceType border) {
    const math::Vector<SceneSpaceType, 3>& front = this->Front();
    const math::Point<SceneSpaceType, 3>& pos = this->Position();
    SceneSpaceType dist, minDist, maxDist;

    dist = front.Dot(bbox.GetLeftBottomBack() - pos);
    minDist = maxDist = dist;

    dist = front.Dot(bbox.GetLeftBottomFront() - pos);
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) maxDist = dist;

    dist = front.Dot(bbox.GetLeftTopBack() - pos);
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) maxDist = dist;

    dist = front.Dot(bbox.GetLeftTopFront() - pos);
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) maxDist = dist;

    dist = front.Dot(bbox.GetRightBottomBack() - pos);
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) maxDist = dist;

    dist = front.Dot(bbox.GetRightBottomFront() - pos);
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) maxDist = dist;

    dist = front.Dot(bbox.GetRightTopBack() - pos);
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) maxDist = dist;

    dist = front.Dot(bbox.GetRightTopFront() - pos);
    if (dist < minDist) minDist = dist;
    if (dist > maxDist) maxDist = dist;

    minDist -= border;
    maxDist += border;

    this->SetClip(minDist, maxDist);

}


/*
 * vislib::graphics::CameraParameters::operator=
 */
vislib::graphics::CameraParameters& 
vislib::graphics::CameraParameters::operator=(
        const vislib::graphics::CameraParameters& rhs) {
    // Intentionally empty
    return *this;
}


/*
 * vislib::graphics::CameraParameters::operator==
 */
bool vislib::graphics::CameraParameters::operator==(
        const vislib::graphics::CameraParameters& rhs) const {
    return (this == &rhs);
}
