/*
 * CameraParameters.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */


#include "vislib/CameraParameters.h"
#include "vislib/mathtypes.h"


/*
 * vislib::graphics::CameraParameters::CameraParameters
 */
vislib::graphics::CameraParameters::CameraParameters(void)
        : vislib::Serialisable() {
}


/* 
 * vislib::graphics::CameraParameters::CameraParameters 
 */
vislib::graphics::CameraParameters::CameraParameters(
        const vislib::graphics::CameraParameters& rhs) 
        : vislib::Serialisable(rhs) {
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
 * vislib::graphics::CameraParameters::Deserialise
 */
void vislib::graphics::CameraParameters::Deserialise(
        vislib::Serialiser& serialiser) {
    float f0, f1, f2, f3, f4, f5, f6, f7, f8;
    INT32 i;
    // TODO: Add name parameters
    serialiser.Deserialise(f0);
    this->SetApertureAngle(f0);
    serialiser.Deserialise(f0);
    serialiser.Deserialise(f1);
    this->SetClip(f0, f1);
    serialiser.Deserialise(i);
    this->SetProjection(static_cast<ProjectionType>(i));
    serialiser.Deserialise(f0);
    serialiser.Deserialise(i);
    serialiser.Deserialise(f1);
    this->SetStereoParameters(f0, static_cast<StereoEye>(i), f1);
    serialiser.Deserialise(f0);
    serialiser.Deserialise(f1);
    serialiser.Deserialise(f2);
    serialiser.Deserialise(f3);
    serialiser.Deserialise(f4);
    serialiser.Deserialise(f5);
    serialiser.Deserialise(f6);
    serialiser.Deserialise(f7);
    serialiser.Deserialise(f8);
    this->SetView(vislib::math::Point<SceneSpaceType, 3>(f0, f1, f2),
        vislib::math::Point<SceneSpaceType, 3>(f3, f4, f5),
        vislib::math::Vector<SceneSpaceType, 3>(f6, f7, f8));
    serialiser.Deserialise(f0);
    serialiser.Deserialise(f1);
    serialiser.Deserialise(f2);
    serialiser.Deserialise(f3);
    serialiser.Deserialise(f4);
    serialiser.Deserialise(f5);
    this->SetVirtualViewSize(f4, f5);
    this->SetTileRect(vislib::math::Rectangle<ImageSpaceType>(f0, f1, f2, f3));
}


/*
 * vislib::graphics::CameraParameters::Serialise
 */
void vislib::graphics::CameraParameters::Serialise(
        vislib::Serialiser& serialiser) const {
    // TODO: Add name parameters
    serialiser.Serialise((float)this->ApertureAngle());
    serialiser.Serialise((float)this->NearClip());
    serialiser.Serialise((float)this->FarClip());
    serialiser.Serialise((INT32)this->Projection());
    serialiser.Serialise((float)this->StereoDisparity());
    serialiser.Serialise((INT32)this->Eye());
    serialiser.Serialise((float)this->FocalDistance());
    serialiser.Serialise((float)this->Position().X());
    serialiser.Serialise((float)this->Position().Y());
    serialiser.Serialise((float)this->Position().Z());
    serialiser.Serialise((float)this->LookAt().X());
    serialiser.Serialise((float)this->LookAt().Y());
    serialiser.Serialise((float)this->LookAt().Z());
    serialiser.Serialise((float)this->Up().X());
    serialiser.Serialise((float)this->Up().Y());
    serialiser.Serialise((float)this->Up().Z());
    serialiser.Serialise((float)this->TileRect().Left());
    serialiser.Serialise((float)this->TileRect().Bottom());
    serialiser.Serialise((float)this->TileRect().Right());
    serialiser.Serialise((float)this->TileRect().Top());
    serialiser.Serialise((float)this->VirtualViewSize().Width());
    serialiser.Serialise((float)this->VirtualViewSize().Height());
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
