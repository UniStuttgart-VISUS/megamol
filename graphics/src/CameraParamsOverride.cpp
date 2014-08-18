/*
 * CameraParamsOverride.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */


#include "vislib/CameraParamsOverride.h"
#include "vislib/assert.h"


/*
 * vislib::graphics::CameraParamsOverride::CameraParamsOverride
 */
vislib::graphics::CameraParamsOverride::CameraParamsOverride(void)
        : CameraParameters(), syncNumberOff(0), base(NULL) {
}


/*
 * vislib::graphics::CameraParamsOverride::CameraParamsOverride
 */
vislib::graphics::CameraParamsOverride::CameraParamsOverride(
        const vislib::SmartPtr<vislib::graphics::CameraParameters>& params) 
        : CameraParameters(), syncNumberOff(0), base(params) {
}


/*
 * vislib::graphics::CameraParamsOverride::~CameraParamsOverride
 */
vislib::graphics::CameraParamsOverride::~CameraParamsOverride(void) {
    this->base = NULL;
}


/*
 * vislib::graphics::CameraParamsOverride::ApplyLimits
 */
void vislib::graphics::CameraParamsOverride::ApplyLimits(void) {
    ASSERT(!this->base.IsNull());
    this->base->ApplyLimits();
}


/*
 * vislib::graphics::CameraParamsOverride::AutoFocusOffset
 */
vislib::graphics::SceneSpaceType
vislib::graphics::CameraParamsOverride::AutoFocusOffset(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->AutoFocusOffset();
}


/*
 * vislib::graphics::CameraParamsOverride::CoordSystemType
 */
vislib::math::CoordSystemType
vislib::graphics::CameraParamsOverride::CoordSystemType(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->CoordSystemType();
}


/*
 * vislib::graphics::CameraParamsOverride::Eye
 */
vislib::graphics::CameraParameters::StereoEye 
vislib::graphics::CameraParamsOverride::Eye(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->Eye();
}


/*
 * vislib::graphics::CameraParamsOverride::EyeDirection
 */
vislib::math::Vector<vislib::graphics::SceneSpaceType, 3> 
vislib::graphics::CameraParamsOverride::EyeDirection(void) const {
    if (this->Projection() != STEREO_TOE_IN) {
        return this->Front();
    } else {
        math::Vector<SceneSpaceType, 3> eyefront = (this->Position()
            + (this->Front() * this->FocalDistance())) 
            - this->EyePosition();
        eyefront.Normalise();
        return eyefront;
    }
}


/*
 * vislib::graphics::CameraParamsOverride::EyeUpVector
 */
vislib::math::Vector<vislib::graphics::SceneSpaceType, 3> 
vislib::graphics::CameraParamsOverride::EyeUpVector(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->EyeUpVector(); // until we get upper and lower eyes
}


/*
 * vislib::graphics::CameraParamsOverride::EyeRightVector
 */
vislib::math::Vector<vislib::graphics::SceneSpaceType, 3> 
vislib::graphics::CameraParamsOverride::EyeRightVector(void) const {
    math::Vector<SceneSpaceType, 3> eyeright 
        = this->EyeDirection().Cross(this->EyeUpVector());
    eyeright.Normalise();
    if (this->CoordSystemType() == math::COORD_SYS_LEFT_HANDED) {
        eyeright *= static_cast<SceneSpaceType>(-1);
    }
    return eyeright;
}


/*
 * vislib::graphics::CameraParamsOverride::EyePosition
 */
vislib::math::Point<vislib::graphics::SceneSpaceType, 3> 
vislib::graphics::CameraParamsOverride::EyePosition(void) const {
    ASSERT(!this->base.IsNull());
    if ((this->Projection() == MONO_PERSPECTIVE) 
            || (this->Projection() == MONO_ORTHOGRAPHIC)) {
        return this->Position();
    } else {
        return this->Position() + (this->Right() 
            * (this->HalfStereoDisparity()
            * ((this->Eye() == RIGHT_EYE)
            ? static_cast<SceneSpaceType>(1) 
            : static_cast<SceneSpaceType>(-1)) ));
    }
}


/*
 * vislib::graphics::CameraParamsOverride::FarClip
 */
vislib::graphics::SceneSpaceType 
vislib::graphics::CameraParamsOverride::FarClip(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->FarClip();
}


/*
 * vislib::graphics::CameraParamsOverride::FocalDistance
 */
vislib::graphics::SceneSpaceType 
vislib::graphics::CameraParamsOverride::FocalDistance(bool autofocus) const {
    ASSERT(!this->base.IsNull());
    return this->base->FocalDistance(autofocus);
}


/*
 * vislib::graphics::CameraParamsOverride::Front
 */
const vislib::math::Vector<vislib::graphics::SceneSpaceType, 3>& 
vislib::graphics::CameraParamsOverride::Front(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->Front();
}


/*
 * vislib::graphics::CameraParamsOverride::HalfApertureAngle
 */
vislib::math::AngleRad 
vislib::graphics::CameraParamsOverride::HalfApertureAngle(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->HalfApertureAngle();
}


/*
 * vislib::graphics::CameraParamsOverride::HalfStereoDisparity
 */
vislib::graphics::SceneSpaceType 
vislib::graphics::CameraParamsOverride::HalfStereoDisparity(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->HalfStereoDisparity();
}


/*
 * vislib::graphics::CameraParamsOverride::IsSimilar
 */
bool vislib::graphics::CameraParamsOverride::IsSimilar(
        const SmartPtr<CameraParameters> rhs) const {
    return this->ParametersTopBase()->IsSimilar(rhs);
}


/*
 * vislib::graphics::CameraParamsOverride::IsValid
 */
bool vislib::graphics::CameraParamsOverride::IsValid(void) const {
    if (!this->base.IsNull()) {
        const CameraParamsOverride *cpo 
            = this->base.DynamicCast<CameraParamsOverride>();
        if (cpo != NULL) {
            return cpo->IsValid();
        }
    }
    return false;
}


/*
 * vislib::graphics::CameraParamsOverride::Limits
 */
vislib::SmartPtr<vislib::graphics::CameraParameterLimits> 
vislib::graphics::CameraParamsOverride::Limits(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->Limits();
}


/*
 * vislib::graphics::CameraParamsOverride::LookAt
 */
const vislib::math::Point<vislib::graphics::SceneSpaceType, 3>& 
vislib::graphics::CameraParamsOverride::LookAt(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->LookAt();
}


/*
 * vislib::graphics::CameraParamsOverride::NearClip
 */
vislib::graphics::SceneSpaceType 
vislib::graphics::CameraParamsOverride::NearClip(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->NearClip();
}


/*
 * vislib::graphics::CameraParamsOverride::ParametersBase
 */
const vislib::SmartPtr<vislib::graphics::CameraParameters>& 
vislib::graphics::CameraParamsOverride::ParametersBase(void) const {
    return this->base;
}


/*
 * vislib::graphics::CameraParamsOverride::ParametersTopBase
 */
const vislib::SmartPtr<vislib::graphics::CameraParameters>& 
vislib::graphics::CameraParamsOverride::ParametersTopBase(void) const {
    if (!this->base.IsNull()) {
        const CameraParamsOverride *cpo 
            = this->base.DynamicCast<CameraParamsOverride>();
        if (cpo != NULL) {
            return cpo->ParametersTopBase();
        } 
    }
    return this->base;
}


/*
 * vislib::graphics::CameraParamsOverride::Position
 */
const vislib::math::Point<vislib::graphics::SceneSpaceType, 3>& 
vislib::graphics::CameraParamsOverride::Position(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->Position();
}


/*
 * vislib::graphics::CameraParamsOverride::Projection
 */
vislib::graphics::CameraParameters::ProjectionType 
vislib::graphics::CameraParamsOverride::Projection(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->Projection();
}


/*
 * vislib::graphics::CameraParamsOverride::Reset
 */
void vislib::graphics::CameraParamsOverride::Reset(void) {
    ASSERT(!this->base.IsNull());
    this->base->Reset();
    this->resetOverride();
}


/*
 * vislib::graphics::CameraParamsOverride::ResetTileRect
 */
void vislib::graphics::CameraParamsOverride::ResetTileRect(void) {
    ASSERT(!this->base.IsNull());
    this->base->ResetTileRect();
}


/*
 * vislib::graphics::CameraParamsOverride::Right
 */
const vislib::math::Vector<vislib::graphics::SceneSpaceType, 3>& 
vislib::graphics::CameraParamsOverride::Right(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->Right();
}


/*
 * vislib::graphics::CameraParamsOverride::SetApertureAngle
 */
void vislib::graphics::CameraParamsOverride::SetApertureAngle(
        vislib::math::AngleDeg apertureAngle) {
    ASSERT(!this->base.IsNull());
    this->base->SetApertureAngle(apertureAngle);
}


/*
 * vislib::graphics::CameraParamsOverride::SetAutoFocusOffset
 */
void vislib::graphics::CameraParamsOverride::SetAutoFocusOffset(
        vislib::graphics::SceneSpaceType offset) {
    ASSERT(!this->base.IsNull());
    this->base->SetAutoFocusOffset(offset);
}


/*
 * vislib::graphics::CameraParamsOverride::SetClip
 */
void vislib::graphics::CameraParamsOverride::SetClip(
        vislib::graphics::SceneSpaceType nearClip, 
        vislib::graphics::SceneSpaceType farClip) {
    ASSERT(!this->base.IsNull());
    this->base->SetClip(nearClip, farClip);
}


/*
 * vislib::graphics::CameraParamsOverride::SetCoordSystemType
 */
void vislib::graphics::CameraParamsOverride::SetCoordSystemType(
        vislib::math::CoordSystemType coordSysType) {
    ASSERT(!this->base.IsNull());
    this->base->SetCoordSystemType(coordSysType);
}


/*
 * vislib::graphics::CameraParamsOverride::SetEye
 */
void vislib::graphics::CameraParamsOverride::SetEye(
        vislib::graphics::CameraParameters::StereoEye eye) {
    ASSERT(!this->base.IsNull());
    this->base->SetEye(eye);
}


/*
 * vislib::graphics::CameraParamsOverride::SetFarClip
 */
void vislib::graphics::CameraParamsOverride::SetFarClip(
        vislib::graphics::SceneSpaceType farClip) {
    ASSERT(!this->base.IsNull());
    this->base->SetFarClip(farClip);
}


/*
 * vislib::graphics::CameraParamsOverride::SetFocalDistance
 */
void vislib::graphics::CameraParamsOverride::SetFocalDistance(
        vislib::graphics::SceneSpaceType focalDistance) {
    ASSERT(!this->base.IsNull());
    this->base->SetFocalDistance(focalDistance);
}


/*
 * vislib::graphics::CameraParamsOverride::SetLimits
 */
void vislib::graphics::CameraParamsOverride::SetLimits(
        const vislib::SmartPtr<vislib::graphics::CameraParameterLimits>& 
        limits) {
    ASSERT(!this->base.IsNull());
    this->base->SetLimits(limits);
}


/*
 * vislib::graphics::CameraParamsOverride::SetLookAt
 */
void vislib::graphics::CameraParamsOverride::SetLookAt(
        const vislib::math::Point<vislib::graphics::SceneSpaceType, 3>& 
        lookAt) {
    ASSERT(!this->base.IsNull());
    this->base->SetLookAt(lookAt);
}


/*
 * vislib::graphics::CameraParamsOverride::SetNearClip
 */
void vislib::graphics::CameraParamsOverride::SetNearClip(
        vislib::graphics::SceneSpaceType nearClip) {
    ASSERT(!this->base.IsNull());
    this->base->SetNearClip(nearClip);
}


/*
 * vislib::graphics::CameraParamsOverride::SetParametersBase
 */
bool vislib::graphics::CameraParamsOverride::SetParametersBase(
        const vislib::SmartPtr<vislib::graphics::CameraParameters>& params) {
    this->preBaseSet(params);
    this->base = params;
    return this->IsValid();
}


/*
 * vislib::graphics::CameraParamsOverride::SetPosition
 */
void vislib::graphics::CameraParamsOverride::SetPosition(
        const vislib::math::Point<vislib::graphics::SceneSpaceType, 3>& 
        position) {
    ASSERT(!this->base.IsNull());
    this->base->SetPosition(position);
}


/*
 * vislib::graphics::CameraParamsOverride::SetProjection
 */
void vislib::graphics::CameraParamsOverride::SetProjection(
        vislib::graphics::CameraParameters::ProjectionType projectionType) {
    ASSERT(!this->base.IsNull());
    this->base->SetProjection(projectionType);
}


/*
 * vislib::graphics::CameraParamsOverride::SetStereoDisparity
 */
void vislib::graphics::CameraParamsOverride::SetStereoDisparity(
        vislib::graphics::SceneSpaceType stereoDisparity) {
    ASSERT(!this->base.IsNull());
    this->base->SetStereoDisparity(stereoDisparity);
}


/*
 * vislib::graphics::CameraParamsOverride::SetStereoParameters
 */
void vislib::graphics::CameraParamsOverride::SetStereoParameters(
        vislib::graphics::SceneSpaceType stereoDisparity, 
        vislib::graphics::CameraParameters::StereoEye eye, 
        vislib::graphics::SceneSpaceType focalDistance) {
    ASSERT(!this->base.IsNull());
    this->base->SetStereoParameters(stereoDisparity, eye, focalDistance);
}


/*
 * vislib::graphics::CameraParamsOverride::SetTileRect
 */
void vislib::graphics::CameraParamsOverride::SetTileRect(
        const vislib::math::Rectangle<vislib::graphics::ImageSpaceType>& 
        tileRect) {
    ASSERT(!this->base.IsNull());
    this->base->SetTileRect(tileRect);
}


/*
 * vislib::graphics::CameraParamsOverride::SetUp
 */
void vislib::graphics::CameraParamsOverride::SetUp(
        const vislib::math::Vector<vislib::graphics::SceneSpaceType, 3>& up) {
    ASSERT(!this->base.IsNull());
    this->base->SetUp(up);
}


/*
 * vislib::graphics::CameraParamsOverride::SetView
 */
void vislib::graphics::CameraParamsOverride::SetView(const 
        vislib::math::Point<vislib::graphics::SceneSpaceType, 3>& position, 
        const vislib::math::Point<vislib::graphics::SceneSpaceType, 3>& lookAt,
        const vislib::math::Vector<vislib::graphics::SceneSpaceType, 3>& up) {
    ASSERT(!this->base.IsNull());
    this->base->SetView(position, lookAt, up);
}


/*
 * vislib::graphics::CameraParamsOverride::SetVirtualViewSize
 */
void vislib::graphics::CameraParamsOverride::SetVirtualViewSize(
        const vislib::math::Dimension<vislib::graphics::ImageSpaceType, 2>& 
        viewSize) {
    ASSERT(!this->base.IsNull());
    this->base->SetVirtualViewSize(viewSize);
}


/*
 * vislib::graphics::CameraParamsOverride::SyncNumber
 */
unsigned int vislib::graphics::CameraParamsOverride::SyncNumber(void) 
        const {
    ASSERT(!this->base.IsNull());
    return this->base->SyncNumber() + this->syncNumberOff;
}


/*
 * vislib::graphics::CameraParamsOverride::TileRect
 */
const vislib::math::Rectangle<vislib::graphics::ImageSpaceType>& 
vislib::graphics::CameraParamsOverride::TileRect(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->TileRect();
}


/*
 * vislib::graphics::CameraParamsOverride::Up
 */
const vislib::math::Vector<vislib::graphics::SceneSpaceType, 3>& 
vislib::graphics::CameraParamsOverride::Up(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->Up();
}


/*
 * vislib::graphics::CameraParamsOverride::VirtualViewSize
 */
const vislib::math::Dimension<vislib::graphics::ImageSpaceType, 2>& 
vislib::graphics::CameraParamsOverride::VirtualViewSize(void) const {
    ASSERT(!this->base.IsNull());
    return this->base->VirtualViewSize();
}


/*
 * vislib::graphics::CameraParamsOverride::operator=
 */
vislib::graphics::CameraParamsOverride& 
vislib::graphics::CameraParamsOverride::operator=(
        const vislib::graphics::CameraParamsOverride& rhs) {
    this->base = rhs.base;
    this->syncNumberOff++;
    return *this;
}


/*
 * vislib::graphics::CameraParamsOverride::operator==
 */
bool vislib::graphics::CameraParamsOverride::operator==(
        const vislib::graphics::CameraParamsOverride& rhs) const {
    return ((this->base == rhs.base) 
        && (this->syncNumberOff == rhs.syncNumberOff));
}


/*
 * vislib::graphics::CameraParamsOverride::indicateValueChange
 */
void vislib::graphics::CameraParamsOverride::indicateValueChange(void) {
    this->syncNumberOff++;
}


/*
 * vislib::graphics::CameraParamsOverride::paramsBase
 */
const vislib::SmartPtr<vislib::graphics::CameraParameters>& 
vislib::graphics::CameraParamsOverride::paramsBase(void) const {
    return this->base;
}
