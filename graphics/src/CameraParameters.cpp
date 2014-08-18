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

    if (!vislib::math::IsEqual(this->NearClip(), minDist)
            || !vislib::math::IsEqual(this->FarClip(), maxDist)) {

        this->SetClip(minDist, maxDist);
    }

}


/*
 * vislib::graphics::CameraParameters::Deserialise
 */
void vislib::graphics::CameraParameters::Deserialise(
        vislib::Serialiser& serialiser) {
    float f0, f1, f2, f3, f4, f5, f6, f7, f8;
    INT32 i;
    // TODO: Add name parameters
    serialiser.Deserialise(f0, "ApertureAngle");
    this->SetApertureAngle(f0);
    serialiser.Deserialise(i, "CoordSystemType");
    this->SetCoordSystemType(static_cast<math::CoordSystemType>(i));
    serialiser.Deserialise(f0, "NearClip");
    serialiser.Deserialise(f1, "FarClip");
    this->SetClip(f0, f1);
    serialiser.Deserialise(i, "Projection");
    this->SetProjection(static_cast<ProjectionType>(i));
    serialiser.Deserialise(f0, "StereoDisparity");
    serialiser.Deserialise(i, "StereoEye");
    serialiser.Deserialise(f1, "FocalDistance");
    this->SetStereoParameters(f0, static_cast<StereoEye>(i), f1);
    serialiser.Deserialise(f0, "PositionX");
    serialiser.Deserialise(f1, "PositionY");
    serialiser.Deserialise(f2, "PositionZ");
    serialiser.Deserialise(f3, "LookAtX");
    serialiser.Deserialise(f4, "LookAtY");
    serialiser.Deserialise(f5, "LookAtZ");
    serialiser.Deserialise(f6, "UpX");
    serialiser.Deserialise(f7, "UpY");
    serialiser.Deserialise(f8, "UpZ");
    this->SetView(vislib::math::Point<SceneSpaceType, 3>(f0, f1, f2),
        vislib::math::Point<SceneSpaceType, 3>(f3, f4, f5),
        vislib::math::Vector<SceneSpaceType, 3>(f6, f7, f8));
    serialiser.Deserialise(f0, "TileLeft");
    serialiser.Deserialise(f1, "TileBottom");
    serialiser.Deserialise(f2, "TileRight");
    serialiser.Deserialise(f3, "TileTop");
    serialiser.Deserialise(f4, "VirtualViewWidth");
    serialiser.Deserialise(f5, "VirtualViewHeight");
    this->SetVirtualViewSize(f4, f5);
    this->SetTileRect(vislib::math::Rectangle<ImageSpaceType>(f0, f1, f2, f3));
    serialiser.Deserialise(f0, "AutoFocusOffset"); // last because it's new
    this->SetAutoFocusOffset(f0);
}


/*
 * vislib::graphics::CameraParameters::Serialise
 */
void vislib::graphics::CameraParameters::Serialise(
        vislib::Serialiser& serialiser) const {
    serialiser.Serialise((float)this->ApertureAngle(), "ApertureAngle");
    serialiser.Serialise((INT32)this->CoordSystemType(), "CoordSystemType");
    serialiser.Serialise((float)this->NearClip(), "NearClip");
    serialiser.Serialise((float)this->FarClip(), "FarClip");
    serialiser.Serialise((INT32)this->Projection(), "Projection");
    serialiser.Serialise((float)this->StereoDisparity(), "StereoDisparity");
    serialiser.Serialise((INT32)this->Eye(), "StereoEye");
    serialiser.Serialise((float)this->FocalDistance(), "FocalDistance");
    serialiser.Serialise((float)this->Position().X(), "PositionX");
    serialiser.Serialise((float)this->Position().Y(), "PositionY");
    serialiser.Serialise((float)this->Position().Z(), "PositionZ");
    serialiser.Serialise((float)this->LookAt().X(), "LookAtX");
    serialiser.Serialise((float)this->LookAt().Y(), "LookAtY");
    serialiser.Serialise((float)this->LookAt().Z(), "LookAtZ");
    serialiser.Serialise((float)this->Up().X(), "UpX");
    serialiser.Serialise((float)this->Up().Y(), "UpY");
    serialiser.Serialise((float)this->Up().Z(), "UpZ");
    serialiser.Serialise((float)this->TileRect().Left(), "TileLeft");
    serialiser.Serialise((float)this->TileRect().Bottom(), "TileBottom");
    serialiser.Serialise((float)this->TileRect().Right(), "TileRight");
    serialiser.Serialise((float)this->TileRect().Top(), "TileTop");
    serialiser.Serialise((float)this->VirtualViewSize().Width(), "VirtualViewWidth");
    serialiser.Serialise((float)this->VirtualViewSize().Height(), "VirtualViewHeight");
    serialiser.Serialise((float)this->AutoFocusOffset(), "AutoFocusOffset"); // last because it's new
}


/*
 * vislib::graphics::CameraParameters::operator=
 */
vislib::graphics::CameraParameters& 
vislib::graphics::CameraParameters::operator=(
        const vislib::graphics::CameraParameters& rhs) {
    if (this != &rhs) {
        this->SetCoordSystemType(rhs.CoordSystemType());
        this->SetClip(rhs.NearClip(), rhs.FarClip());
        this->SetApertureAngle(rhs.ApertureAngle());
        this->SetAutoFocusOffset(rhs.AutoFocusOffset());
        this->SetProjection(rhs.Projection());
        this->SetStereoParameters(rhs.StereoDisparity(), rhs.Eye(), rhs.FocalDistance());
        this->SetView(rhs.Position(), rhs.LookAt(), rhs.Up());
        this->SetTileRect(rhs.TileRect());
        this->SetVirtualViewSize(rhs.VirtualViewSize());
        this->SetLimits(rhs.Limits());
    }
    return *this;
}


/*
 * vislib::graphics::CameraParameters::operator==
 */
bool vislib::graphics::CameraParameters::operator==(
        const vislib::graphics::CameraParameters& rhs) const {
    return (this == &rhs);
}
