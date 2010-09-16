/*
 * CameraParamsStore.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#include "vislib/CameraParamsStore.h"
#include "vislib/mathtypes.h"


/* Default values */
#define DEFAULT_AUTO_FOCUS_OFFSET 0.0f
#define DEFAULT_COORD_SYS_TYPE    math::COORD_SYS_RIGHT_HANDED
#define DEFAULT_EYE               LEFT_EYE
#define DEFAULT_FAR_CLIP          10.0f
#define DEFAULT_FOCAL_DIST        0.0f
#define DEFAULT_FRONT             math::Vector<float, 3>(0.0f, 0.0f, -1.0f)
#define DEFAULT_HALF_AP_ANGLE     0.5236f
#define DEFAULT_HALF_DISPARITY    0.125f
#define DEFAULT_LOOK_AT           math::Point<float, 3>(0.0f, 0.0f, 0.0f)
#define DEFAULT_NEAR_CLIP         0.1f
#define DEFAULT_POSITION          math::Point<float, 3>(0.0f, 0.0f, 1.0f)
#define DEFAULT_PROJECTION_TYPE   MONO_PERSPECTIVE
#define DEFAULT_RIGHT             math::Vector<float, 3>(1.0f, 0.0f, 0.0f)
#define DEFAULT_UP                math::Vector<float, 3>(0.0f, 1.0f, 0.0f)
#define DEFAULT_VIEW_SIZE         math::Dimension<float, 2>(100.0f, 100.0f)
#define DEFAULT_TILE_RECT         math::Rectangle<float>(math::Point<float, 2>\
                                  (0.0f, 0.0f), DEFAULT_VIEW_SIZE)


/*
 * vislib::graphics::CameraParamsStore::CameraParamsStore
 */
vislib::graphics::CameraParamsStore::CameraParamsStore(void) 
        : CameraParameters(), autoFocusOffset(DEFAULT_AUTO_FOCUS_OFFSET),
        coordSysType(DEFAULT_COORD_SYS_TYPE), eye(DEFAULT_EYE),
        farClip(DEFAULT_FAR_CLIP), focalDistance(DEFAULT_FOCAL_DIST),
        front(DEFAULT_FRONT), halfApertureAngle(DEFAULT_HALF_AP_ANGLE),
        halfStereoDisparity(DEFAULT_HALF_DISPARITY), 
        limits(CameraParameterLimits::DefaultLimits()), 
        lookAt(DEFAULT_LOOK_AT), nearClip(DEFAULT_NEAR_CLIP), 
        position(DEFAULT_POSITION), projectionType(DEFAULT_PROJECTION_TYPE), 
        right(DEFAULT_RIGHT), syncNumber(0), tileRect(DEFAULT_TILE_RECT), 
        up(DEFAULT_UP), virtualViewSize(DEFAULT_VIEW_SIZE) {
}


/* 
 * vislib::graphics::CameraParamsStore::CameraParamsStore 
 */
vislib::graphics::CameraParamsStore::CameraParamsStore(
        const vislib::graphics::CameraParamsStore& rhs) 
        : CameraParameters(), syncNumber(0) {
    *this = rhs;
}


/* 
 * vislib::graphics::CameraParamsStore::CameraParamsStore 
 */
vislib::graphics::CameraParamsStore::CameraParamsStore(
        const vislib::graphics::CameraParameters& rhs) 
        : CameraParameters(), syncNumber(0) {
    *this = rhs;
}


/*
 * vislib::graphics::CameraParamsStore::~CameraParamsStore
 */
vislib::graphics::CameraParamsStore::~CameraParamsStore(void) {
}


/*
 * vislib::graphics::CameraParamsStore::ApplyLimits
 */
void vislib::graphics::CameraParamsStore::ApplyLimits(void) {
    // copy parameters to avoid aliasing
    math::AngleRad aa = this->ApertureAngle();
    SceneSpaceType nc = this->NearClip();
    SceneSpaceType fc = this->FarClip();
    SceneSpaceType sd = this->StereoDisparity();
    StereoEye e = this->Eye();
    SceneSpaceType fd = this->FocalDistance();
    math::Point<SceneSpaceType, 3> p = this->Position();
    math::Point<SceneSpaceType, 3> la = this->LookAt();
    math::Vector<SceneSpaceType, 3> up = this->Up();

    this->SetApertureAngle(aa);
    this->SetClip(nc, fc);
    this->SetStereoParameters(sd, e, fd);
    this->SetView(p, la, up);
}


/*
 * vislib::graphics::CameraParamsStore::AutoFocusOffset
 */
vislib::graphics::SceneSpaceType
vislib::graphics::CameraParamsStore::AutoFocusOffset(void) const {
    return this->autoFocusOffset;
}


/*
 * vislib::graphics::CameraParamsStore::CoordSystemType
 */
vislib::math::CoordSystemType
vislib::graphics::CameraParamsStore::CoordSystemType(void) const {
    return this->coordSysType;
}


/*
 * vislib::graphics::CameraParamsStore::Eye
 */
vislib::graphics::CameraParameters::StereoEye 
vislib::graphics::CameraParamsStore::Eye(void) const {
    return this->eye;
}


/*
 * vislib::graphics::CameraParamsStore::EyeDirection
 */
vislib::math::Vector<vislib::graphics::SceneSpaceType, 3> 
vislib::graphics::CameraParamsStore::EyeDirection(void) const {
    if (this->projectionType != STEREO_TOE_IN) {
        return this->front;
    } else {
        math::Vector<SceneSpaceType, 3> eyefront = (this->position
            + (this->front * this->focalDistance)) - this->EyePosition();
        eyefront.Normalise();
        return eyefront;
    }
}


/*
 * vislib::graphics::CameraParamsEyeOverride::EyeUpVector
 */
vislib::math::Vector<vislib::graphics::SceneSpaceType, 3> 
vislib::graphics::CameraParamsStore::EyeUpVector(void) const {
    return this->up; // until we get upper and lower eyes
}


/*
 * vislib::graphics::CameraParamsEyeOverride::EyeRightVector
 */
vislib::math::Vector<vislib::graphics::SceneSpaceType, 3> 
vislib::graphics::CameraParamsStore::EyeRightVector(void) const {
    math::Vector<SceneSpaceType, 3> eyeright 
        = this->EyeDirection().Cross(this->EyeUpVector());
    eyeright.Normalise();
    if (this->CoordSystemType() == math::COORD_SYS_LEFT_HANDED) {
        eyeright *= static_cast<SceneSpaceType>(-1);
    }
    return eyeright;
}


/*
 * vislib::graphics::CameraParamsStore::EyePosition
 */
vislib::math::Point<vislib::graphics::SceneSpaceType, 3> 
vislib::graphics::CameraParamsStore::EyePosition(void) const {
    if ((this->projectionType == MONO_PERSPECTIVE) 
            || (this->projectionType == MONO_ORTHOGRAPHIC)) {
        return this->position;
    } else {
        return this->position + (this->right * (this->halfStereoDisparity
            * ((this->Eye() == RIGHT_EYE)
            ? static_cast<SceneSpaceType>(1) 
            : static_cast<SceneSpaceType>(-1)) ));
    }
}


/*
 * vislib::graphics::CameraParamsStore::FarClip
 */
vislib::graphics::SceneSpaceType vislib::graphics::CameraParamsStore::FarClip(
        void) const {
    return this->farClip;
}


/*
 * vislib::graphics::CameraParamsStore::FocalDistance
 */
vislib::graphics::SceneSpaceType 
vislib::graphics::CameraParamsStore::FocalDistance(bool autofocus) const {
     /* no epsilon test is needed, since this is an explicity set symbol */
    if (autofocus && (this->focalDistance == 0.0f)) {
        // autofocus
        return this->LookAt().Distance(this->Position())
            + this->autoFocusOffset;
    }
    return this->focalDistance;
}


/*
 * vislib::graphics::CameraParamsStore::Front
 */
const vislib::math::Vector<vislib::graphics::SceneSpaceType, 3>& 
vislib::graphics::CameraParamsStore::Front(void) const {
    return this->front;
}


/*
 * vislib::graphics::CameraParamsStore::HalfApertureAngle
 */
vislib::math::AngleRad 
vislib::graphics::CameraParamsStore::HalfApertureAngle(void) const {
    return this->halfApertureAngle;
}


/*
 * vislib::graphics::CameraParamsStore::HalfStereoDisparity
 */
vislib::graphics::SceneSpaceType 
vislib::graphics::CameraParamsStore::HalfStereoDisparity(void) const {
    return this->halfStereoDisparity;
}


/*
 * vislib::graphics::CameraParamsStore::IsSimilar
 */
bool vislib::graphics::CameraParamsStore::IsSimilar(
        const SmartPtr<CameraParameters> rhs) const {
    if (rhs.DynamicCast<CameraParamsStore>() == NULL) {
        return rhs->IsSimilar(const_cast<CameraParamsStore *>(this));
    } else {
        return rhs.DynamicCast<CameraParamsStore>() == this;
    }
}


/*
 * vislib::graphics::CameraParamsStore::Limits
 */
vislib::SmartPtr<vislib::graphics::CameraParameterLimits> 
vislib::graphics::CameraParamsStore::Limits(void) const {
    return this->limits;
}


/*
 * vislib::graphics::CameraParamsStore::LookAt
 */
const vislib::math::Point<vislib::graphics::SceneSpaceType, 3>& 
vislib::graphics::CameraParamsStore::LookAt(void) const {
    return this->lookAt;
}


/*
 * vislib::graphics::CameraParamsStore::NearClip
 */
vislib::graphics::SceneSpaceType vislib::graphics::CameraParamsStore::NearClip(
        void) const {
    return this->nearClip;
}


/*
 * vislib::graphics::CameraParamsStore::Position
 */
const vislib::math::Point<vislib::graphics::SceneSpaceType, 3>& 
vislib::graphics::CameraParamsStore::Position(void) const {
    return this->position;
}


/*
 * vislib::graphics::CameraParamsStore::Projection
 */
vislib::graphics::CameraParameters::ProjectionType 
vislib::graphics::CameraParamsStore::Projection(void) const {
    return this->projectionType;
}


/*
 * vislib::graphics::CameraParamsStore::Reset
 */
void vislib::graphics::CameraParamsStore::Reset(void) {
    this->autoFocusOffset = DEFAULT_AUTO_FOCUS_OFFSET;
    this->coordSysType = DEFAULT_COORD_SYS_TYPE;
    this->eye = DEFAULT_EYE;
    this->farClip = DEFAULT_FAR_CLIP;
    this->focalDistance = DEFAULT_FOCAL_DIST;
    this->front = DEFAULT_FRONT;
    this->halfApertureAngle = DEFAULT_HALF_AP_ANGLE;
    this->halfStereoDisparity = DEFAULT_HALF_DISPARITY;
    this->limits = CameraParameterLimits::DefaultLimits();
    this->lookAt = DEFAULT_LOOK_AT;
    this->nearClip = DEFAULT_NEAR_CLIP;
    this->position = DEFAULT_POSITION;
    this->projectionType = DEFAULT_PROJECTION_TYPE;
    this->right = DEFAULT_RIGHT;
    this->syncNumber++; // Do not reset syncNumber but indicate the change!
    this->tileRect = DEFAULT_TILE_RECT;
    this->up = DEFAULT_UP;
    this->virtualViewSize = DEFAULT_VIEW_SIZE;
}


/*
 * vislib::graphics::CameraParamsStore::ResetTileRect
 */
void vislib::graphics::CameraParamsStore::ResetTileRect(void) {
    this->tileRect.SetNull();
    this->tileRect.SetSize(this->virtualViewSize);
    this->syncNumber++;
}


/*
 * vislib::graphics::CameraParamsStore::Right
 */
const vislib::math::Vector<vislib::graphics::SceneSpaceType, 3>& 
vislib::graphics::CameraParamsStore::Right(void) const {
    return this->right;
}


/*
 * vislib::graphics::CameraParamsStore::SetApertureAngle
 */
void vislib::graphics::CameraParamsStore::SetApertureAngle(
        vislib::math::AngleDeg apertureAngle) {
    ASSERT(!this->limits.IsNull());

    apertureAngle = math::AngleDeg2Rad(apertureAngle);

    if (this->limits->MinApertureAngle() > apertureAngle) {
        apertureAngle = this->limits->MinApertureAngle();
    } else if (this->limits->MaxApertureAngle() < apertureAngle) {
        apertureAngle = this->limits->MaxApertureAngle();
    }
    this->halfApertureAngle = apertureAngle * 0.5f;
    this->syncNumber++;

}


/*
 * vislib::graphics::CameraParamsStore::SetAutoFocusOffset
 */
void vislib::graphics::CameraParamsStore::SetAutoFocusOffset(
        vislib::graphics::SceneSpaceType offset) {
    this->autoFocusOffset = offset;
}


/*
 * vislib::graphics::CameraParamsStore::SetClip
 */
void vislib::graphics::CameraParamsStore::SetClip(
        vislib::graphics::SceneSpaceType nearClip, 
        vislib::graphics::SceneSpaceType farClip) {
    ASSERT(!this->limits.IsNull());

    if (nearClip < this->limits->MinNearClipDist()) {
        nearClip = this->limits->MinNearClipDist();
    }

    if (farClip < nearClip + this->limits->MinClipPlaneDist()) {
        farClip = nearClip + this->limits->MinClipPlaneDist();
    }

    this->nearClip = nearClip;
    this->farClip = farClip;

    this->syncNumber++;
}


/*
 * vislib::graphics::CameraParamsStore::SetCoordSystemType
 */
void vislib::graphics::CameraParamsStore::SetCoordSystemType(
        vislib::math::CoordSystemType coordSysType) {
    if (this->coordSysType != coordSysType) {
        this->coordSysType = coordSysType;
        this->right *= static_cast<SceneSpaceType>(-1);
        this->syncNumber++;
    }
}


/*
 * vislib::graphics::CameraParamsStore::SetEye
 */
void vislib::graphics::CameraParamsStore::SetEye(
        vislib::graphics::CameraParameters::StereoEye eye) {
    this->eye = eye; // no need to adjust anything else
    this->syncNumber++;
}


/*
 * vislib::graphics::CameraParamsStore::SetFarClip
 */
void vislib::graphics::CameraParamsStore::SetFarClip(
        vislib::graphics::SceneSpaceType farClip) {
    ASSERT(!this->limits.IsNull());

    if (this->nearClip < this->limits->MinNearClipDist()) {
        this->nearClip = this->limits->MinNearClipDist();
    }

    if (farClip < this->nearClip + this->limits->MinClipPlaneDist()) {
        farClip = this->nearClip + this->limits->MinClipPlaneDist();
    }

    this->farClip = farClip;

    this->syncNumber++;
}


/*
 * vislib::graphics::CameraParamsStore::SetFocalDistance
 */
void vislib::graphics::CameraParamsStore::SetFocalDistance(
        vislib::graphics::SceneSpaceType focalDistance) {
    ASSERT(!this->limits.IsNull());

    if (vislib::math::IsEqual(focalDistance, 0.0f)) {
        this->focalDistance = 0.0f; // special indication
    } else {
        if (this->limits->MinFocalDist() > focalDistance) {
            focalDistance = this->limits->MinFocalDist();
        }
        this->focalDistance = focalDistance;
    }
    this->syncNumber++;
}


/*
 * vislib::graphics::CameraParamsStore::SetLimits
 */
void vislib::graphics::CameraParamsStore::SetLimits(
        const vislib::SmartPtr<vislib::graphics::CameraParameterLimits>& limits) {
    if (!limits.IsNull()) {
        this->limits = limits;
        this->ApplyLimits(); // this will also increase syncNumber
    }
}


/*
 * vislib::graphics::CameraParamsStore::SetLookAt
 */
void vislib::graphics::CameraParamsStore::SetLookAt(const 
        vislib::math::Point<vislib::graphics::SceneSpaceType, 3>& lookAt) {
    this->SetView(this->position, lookAt, this->up);
    // syncNumber increase per SFX
}


/*
 * vislib::graphics::CameraParamsStore::SetNearClip
 */
void vislib::graphics::CameraParamsStore::SetNearClip(
        vislib::graphics::SceneSpaceType nearClip) {
    ASSERT(!this->limits.IsNull());

    if (nearClip + this->limits->MinNearClipDist() > this->farClip) {
        nearClip = this->farClip - this->limits->MinNearClipDist();
    }

    if (this->limits->MinNearClipDist() > nearClip) {
        nearClip = this->limits->MinNearClipDist();
    } 

    this->nearClip = nearClip;

    if (this->nearClip + this->limits->MinNearClipDist() > this->farClip) {
        this->farClip = this->nearClip + this->limits->MinNearClipDist();
    }

    this->syncNumber++;
}


/*
 * vislib::graphics::CameraParamsStore::SetPosition
 */
void vislib::graphics::CameraParamsStore::SetPosition(const 
        vislib::math::Point<vislib::graphics::SceneSpaceType, 3>& position) {
    this->SetView(position, this->lookAt, this->up);
    // syncNumber increase per SFX
}


/*
 * vislib::graphics::CameraParamsStore::SetProjection
 */
void vislib::graphics::CameraParamsStore::SetProjection(
        vislib::graphics::CameraParameters::ProjectionType projectionType) {
    this->projectionType = projectionType; // no need to adjust anything else
    this->syncNumber++;
}


/*
 * vislib::graphics::CameraParamsStore::SetStereoDisparity
 */
void vislib::graphics::CameraParamsStore::SetStereoDisparity(
        vislib::graphics::SceneSpaceType stereoDisparity) {
    // This also allows negative values *wtf*
    // no need to adjust anything else
    this->halfStereoDisparity = stereoDisparity * 0.5f; 
    this->syncNumber++;
}


/*
 * vislib::graphics::CameraParamsStore::SetStereoParameters
 */
void vislib::graphics::CameraParamsStore::SetStereoParameters(
        vislib::graphics::SceneSpaceType stereoDisparity, 
        vislib::graphics::CameraParamsStore::StereoEye eye, 
        vislib::graphics::SceneSpaceType focalDistance) {
    ASSERT(!this->limits.IsNull());

    this->halfStereoDisparity = stereoDisparity * 0.5f; 
    this->eye = eye;
    if (vislib::math::IsEqual(focalDistance, 0.0f)) {
        this->focalDistance = 0.0f; // special indication
    } else {
        if (this->limits->MinFocalDist() > focalDistance) {
            focalDistance = this->limits->MinFocalDist();
        }
        this->focalDistance = focalDistance;
    }
    this->syncNumber++;
}


/*
 * vislib::graphics::CameraParamsStore::SetTileRect
 */
void vislib::graphics::CameraParamsStore::SetTileRect(const 
        vislib::math::Rectangle<vislib::graphics::ImageSpaceType>& tileRect) {    
    this->tileRect = tileRect; // no need to adjust anything else
    this->syncNumber++;
}


/*
 * vislib::graphics::CameraParamsStore::SetUp
 */
void vislib::graphics::CameraParamsStore::SetUp(const 
        vislib::math::Vector<vislib::graphics::SceneSpaceType, 3>& up) {
    ASSERT(!this->limits.IsNull());

    if (this->front.IsParallel(up)) {
        // up and directon are parallel
        return;
    }
    
    this->right = this->front.Cross(up);
    this->right.Normalise();
    this->up = this->right.Cross(this->front);
    this->up.Normalise();
    if (this->CoordSystemType() == math::COORD_SYS_LEFT_HANDED) {
        this->right *= static_cast<SceneSpaceType>(-1);
    }

    this->syncNumber++;
}


/*
 * vislib::graphics::CameraParamsStore::SetView
 */
void vislib::graphics::CameraParamsStore::SetView(const 
        vislib::math::Point<vislib::graphics::SceneSpaceType, 3>& position, 
        const vislib::math::Point<vislib::graphics::SceneSpaceType, 3>& lookAt,
        const vislib::math::Vector<vislib::graphics::SceneSpaceType, 3>& up) {
    ASSERT(!this->limits.IsNull());

    math::Vector<SceneSpaceType, 3> dir = lookAt - position;
    SceneSpaceType dirLen = dir.Length();

    if (dir.IsParallel(up)) {
        // up and directon are parallel
        return;
    }
    
    if (dirLen < math::FLOAT_EPSILON) { 
        // lookAt and position are on the same point
        return;
    }

    this->position = position;
    this->front = dir / dirLen;

    if (dirLen < this->limits->MinLookAtDist()) {
        // position and lookAt are too close => move lookAt
        dir = this->front * this->limits->MinLookAtDist();
    }

    this->lookAt = this->position + dir;
    this->right = this->front.Cross(up);
    this->right.Normalise();
    this->up = this->right.Cross(this->front);
    this->up.Normalise(); // should not be neccessary, but to be sure (inaccuracy)
    if (this->CoordSystemType() == math::COORD_SYS_LEFT_HANDED) {
        this->right *= static_cast<SceneSpaceType>(-1);
    }

    this->syncNumber++;
}


/*
 * vislib::graphics::CameraParamsStore::SetVirtualViewSize
 */
void vislib::graphics::CameraParamsStore::SetVirtualViewSize(const 
        vislib::math::Dimension<vislib::graphics::ImageSpaceType, 2>& 
        viewSize) {
    if (math::IsEqual(this->tileRect.GetLeft(), 0.0f)
        && math::IsEqual(this->tileRect.GetBottom(), 0.0f)
        && math::IsEqual(this->tileRect.GetRight(), 
            this->virtualViewSize.Width())
        && math::IsEqual(this->tileRect.GetTop(), 
            this->virtualViewSize.Height())) {
        this->tileRect.SetSize(viewSize);
    }
    this->virtualViewSize = viewSize;
    this->syncNumber++;
}


/*
 * vislib::graphics::CameraParamsStore::SyncNumber
 */
unsigned int vislib::graphics::CameraParamsStore::SyncNumber(void) const {
    return this->syncNumber;
}


/*
 * vislib::graphics::CameraParamsStore::TileRect
 */
const vislib::math::Rectangle<vislib::graphics::ImageSpaceType>& 
vislib::graphics::CameraParamsStore::TileRect(void) const {
    return this->tileRect;
}


/*
 * vislib::graphics::CameraParamsStore::Up
 */
const vislib::math::Vector<vislib::graphics::SceneSpaceType, 3>& 
vislib::graphics::CameraParamsStore::Up(void) const {
    return this->up;
}


/*
 * vislib::graphics::CameraParamsStore::VirtualViewSize
 */
const vislib::math::Dimension<vislib::graphics::ImageSpaceType, 2>& 
vislib::graphics::CameraParamsStore::VirtualViewSize(void) const {
    return this->virtualViewSize;
}


/*
 * vislib::graphics::CameraParamsStore::operator=
 */
vislib::graphics::CameraParamsStore& 
vislib::graphics::CameraParamsStore::operator=(
        const vislib::graphics::CameraParamsStore& rhs) {
    this->autoFocusOffset = rhs.autoFocusOffset;
    this->coordSysType = rhs.coordSysType;
    this->eye = rhs.eye;
    this->farClip = rhs.farClip;
    this->focalDistance = rhs.focalDistance;
    this->front = rhs.front;
    this->halfApertureAngle = rhs.halfApertureAngle;
    this->halfStereoDisparity = rhs.halfStereoDisparity;
    this->limits = rhs.limits; // we dont have to apply limits because rhs is
                               // using the same limits as we will from now on.
    this->lookAt = rhs.lookAt;
    this->nearClip = rhs.nearClip;
    this->position = rhs.position;
    this->projectionType = rhs.projectionType;
    this->right = rhs.right;
    this->syncNumber++; // Do not copy syncNumber but indicate the change!
    this->tileRect = rhs.tileRect;
    this->up = rhs.up;
    this->virtualViewSize = rhs.virtualViewSize;

    return *this;
}


/*
 * vislib::graphics::CameraParamsStore::operator=
 */
vislib::graphics::CameraParamsStore& 
vislib::graphics::CameraParamsStore::operator=(
        const vislib::graphics::CameraParameters& rhs) {
    this->autoFocusOffset = rhs.AutoFocusOffset();
    this->coordSysType = rhs.CoordSystemType();
    this->eye = rhs.Eye();
    this->farClip = rhs.FarClip();
    this->focalDistance = rhs.FocalDistance(false);
    this->front = rhs.Front();
    this->halfApertureAngle = rhs.HalfApertureAngle();
    this->halfStereoDisparity = rhs.HalfStereoDisparity();
    this->lookAt = rhs.LookAt();
    this->nearClip = rhs.NearClip();
    this->position = rhs.Position();
    this->projectionType = rhs.Projection();
    this->right = rhs.Right();
    this->syncNumber++; // Do not copy syncNumber but indicate the change!
    this->tileRect = rhs.TileRect();
    this->up = rhs.Up();
    this->virtualViewSize = rhs.VirtualViewSize();

    this->SetLimits(rhs.Limits());

    return *this;
}


/*
 * vislib::graphics::CameraParamsStore::operator==
 */
bool vislib::graphics::CameraParamsStore::operator==(
        const vislib::graphics::CameraParamsStore& rhs) const {
    return ((this->autoFocusOffset == rhs.autoFocusOffset)
        && (this->coordSysType == rhs.coordSysType)
        && (this->eye == rhs.eye)
        && (this->farClip == rhs.farClip)
        && (this->focalDistance == rhs.focalDistance)
        && (this->front == rhs.front)
        && (this->halfApertureAngle == rhs.halfApertureAngle)
        && (this->halfStereoDisparity == rhs.halfStereoDisparity)
        && (this->limits == rhs.limits)
        && (this->lookAt == rhs.lookAt)
        && (this->nearClip == rhs.nearClip)
        && (this->position == rhs.position)
        && (this->projectionType == rhs.projectionType)
        && (this->right == rhs.right)
        && (this->tileRect == rhs.tileRect)
        && (this->up == rhs.up)
        && (this->virtualViewSize == rhs.virtualViewSize));
}
