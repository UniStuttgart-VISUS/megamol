/*
 * ObservableCameraParams.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/ObservableCameraParams.h"

#include "vislib/assert.h"
#include "vislib/CameraParamsStore.h"
#include "vislib/Trace.h"


/*
 * vislib::graphics::ObservableCameraParams::ObservableCameraParams
 */
vislib::graphics::ObservableCameraParams::ObservableCameraParams(void) 
        : Super(), dirtyFields(0), isBatchInteraction(false), 
        observed(new CameraParamsStore()), isSuspendFire(false) {
    // No one can have registered here, so it is unnecessary to fire the event.
}


/*
 * vislib::graphics::ObservableCameraParams::ObservableCameraParams
 */
vislib::graphics::ObservableCameraParams::ObservableCameraParams(
        SmartPtr<CameraParameters>& observed) 
        : Super(), dirtyFields(0), isBatchInteraction(false), 
        observed(observed), isSuspendFire(false) {
    // No one can have registered here, so it is unnecessary to fire the event.
}


/*
 * vislib::graphics::ObservableCameraParams::ObservableCameraParams
 */
vislib::graphics::ObservableCameraParams::ObservableCameraParams(
        const ObservableCameraParams& rhs) 
        : Super(rhs), dirtyFields(0), isBatchInteraction(false), 
        observed(rhs.observed), isSuspendFire(false) {
    // No one can have registered here, so it is unnecessary to fire the event.
}


/*
 * vislib::graphics::ObservableCameraParams::~ObservableCameraParams
 */
vislib::graphics::ObservableCameraParams::~ObservableCameraParams(
        void) {
    // Nothing to do.
}


/*
 * vislib::graphics::ObservableCameraParams::AddCameraParameterObserver
 */
void vislib::graphics::ObservableCameraParams::AddCameraParameterObserver(
        CameraParameterObserver *observer) {
    ASSERT(observer != NULL);

    if ((observer != NULL) && !this->camParamObservers.Contains(observer)) {
        this->camParamObservers.Append(observer);
    }
}


/*
 * vislib::graphics::ObservableCameraParams::ApplyLimits
 */
void vislib::graphics::ObservableCameraParams::ApplyLimits(void) {
    this->suspendFire();
    this->observed->ApplyLimits();
    this->resumeFire();
    this->fireChanged();
}


/*
 * vislib::graphics::ObservableCameraParams::AutoFocusOffset
 */
vislib::graphics::SceneSpaceType
vislib::graphics::ObservableCameraParams::AutoFocusOffset(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->AutoFocusOffset();
}


/*
 * vislib::graphics::ObservableCameraParams::BeginBatchInteraction
 */
void vislib::graphics::ObservableCameraParams::BeginBatchInteraction(void) {
    this->isBatchInteraction = true;
}


/*
 * vislib::graphics::ObservableCameraParams::CoordSystemType
 */
vislib::math::CoordSystemType
vislib::graphics::ObservableCameraParams::CoordSystemType(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->CoordSystemType();
}


/*
 * vislib::graphics::ObservableCameraParams::EndBatchInteraction
 */
void vislib::graphics::ObservableCameraParams::EndBatchInteraction(void) {
    if (this->isBatchInteraction) {
        this->isBatchInteraction = false;
        this->fireChanged();
    }
}

/*
 * vislib::graphics::ObservableCameraParams::Eye
 */
vislib::graphics::CameraParameters::StereoEye 
vislib::graphics::ObservableCameraParams::Eye(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->Eye();
}


/*
 * vislib::graphics::ObservableCameraParams::EyeDirection
 */
vislib::math::Vector<vislib::graphics::SceneSpaceType, 3> 
vislib::graphics::ObservableCameraParams::EyeDirection(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->EyeDirection();
}


/*
 * vislib::graphics::ObservableCameraParams::EyeUpVector
 */
vislib::math::Vector<vislib::graphics::SceneSpaceType, 3> 
vislib::graphics::ObservableCameraParams::EyeUpVector(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->EyeUpVector();
}


/*
 * vislib::graphics::ObservableCameraParams::EyeRightVector
 */
vislib::math::Vector<vislib::graphics::SceneSpaceType, 3> 
vislib::graphics::ObservableCameraParams::EyeRightVector(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->EyeRightVector();
}


/*
 * vislib::graphics::ObservableCameraParams::EyePosition
 */
vislib::math::Point<vislib::graphics::SceneSpaceType, 3> 
vislib::graphics::ObservableCameraParams::EyePosition(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->EyePosition();
}


/*
 * vislib::graphics::ObservableCameraParams::FarClip
 */
vislib::graphics::SceneSpaceType 
vislib::graphics::ObservableCameraParams::FarClip(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->FarClip();
}


/*
 * vislib::graphics::ObservableCameraParams::FocalDistance
 */
vislib::graphics::SceneSpaceType 
vislib::graphics::ObservableCameraParams::FocalDistance(bool autofocus) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->FocalDistance(autofocus);
}


/*
 * vislib::graphics::ObservableCameraParams::Front
 */
const vislib::math::Vector<vislib::graphics::SceneSpaceType, 3>& 
vislib::graphics::ObservableCameraParams::Front(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->Front();
}


/*
 * vislib::graphics::ObservableCameraParams::HalfApertureAngle
 */
vislib::math::AngleRad 
vislib::graphics::ObservableCameraParams::HalfApertureAngle(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->HalfApertureAngle();
}


/*
 * vislib::graphics::ObservableCameraParams::HalfStereoDisparity
 */
vislib::graphics::SceneSpaceType 
vislib::graphics::ObservableCameraParams::HalfStereoDisparity(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->HalfStereoDisparity();
}


/*
 * vislib::graphics::ObservableCameraParams::Limits
 */
vislib::SmartPtr<vislib::graphics::CameraParameterLimits> 
vislib::graphics::ObservableCameraParams::Limits(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->Limits();
}


/*
 * vislib::graphics::ObservableCameraParams::IsSimilar
 */
bool vislib::graphics::ObservableCameraParams::IsSimilar(
        const SmartPtr<CameraParameters> rhs) const {
    ASSERT(!this->observed.IsNull());
// TODO: MUST IMPLEMENT THIS!
//    return this->observed->IsSimilar(rhs);
    return true;
}


/*
 * vislib::graphics::ObservableCameraParams::LookAt
 */
const vislib::math::Point<vislib::graphics::SceneSpaceType, 3>& 
vislib::graphics::ObservableCameraParams::LookAt(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->LookAt();
}


/*
 * vislib::graphics::ObservableCameraParams::NearClip
 */
vislib::graphics::SceneSpaceType 
vislib::graphics::ObservableCameraParams::NearClip(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->NearClip();
}


/*
 * vislib::graphics::ObservableCameraParams::Position
 */
const vislib::math::Point<vislib::graphics::SceneSpaceType, 3>& 
vislib::graphics::ObservableCameraParams::Position(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->Position();
}


/*
 * vislib::graphics::ObservableCameraParams::Projection
 */
vislib::graphics::CameraParameters::ProjectionType 
vislib::graphics::ObservableCameraParams::Projection(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->Projection();
}


/*
 * vislib::graphics::ObservableCameraParams::RemoveCameraParameterObserver
 */
void vislib::graphics::ObservableCameraParams::RemoveCameraParameterObserver(
        CameraParameterObserver *observer) {
    ASSERT(observer != NULL);
    this->camParamObservers.RemoveAll(observer);
}


/*
 * vislib::graphics::ObservableCameraParams::Reset
 */
void vislib::graphics::ObservableCameraParams::Reset(void) {
    this->suspendFire();
    this->observed->Reset();
    this->resumeFire();
    this->fireChanged();
}


/*
 * vislib::graphics::ObservableCameraParams::ResetTileRect
 */
void vislib::graphics::ObservableCameraParams::ResetTileRect(void) {
    this->suspendFire();
    this->observed->ResetTileRect();
    this->resumeFire();
    this->fireChanged(DIRTY_TILERECT, false);
}


/*
 * vislib::graphics::ObservableCameraParams::Right
 */
const vislib::math::Vector<vislib::graphics::SceneSpaceType, 3>& 
vislib::graphics::ObservableCameraParams::Right(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->Right();
}


/*
 * vislib::graphics::ObservableCameraParams::SetApertureAngle
 */
void vislib::graphics::ObservableCameraParams::SetApertureAngle(
        math::AngleDeg apertureAngle) {
    this->suspendFire();
    this->observed->SetApertureAngle(apertureAngle);
    this->resumeFire();
    this->fireChanged(DIRTY_APERTUREANGLE, false);
}


/*
 * vislib::graphics::ObservableCameraParams::SetAutoFocusOffset
 */
void vislib::graphics::ObservableCameraParams::SetAutoFocusOffset(
        vislib::graphics::SceneSpaceType offset) {
    this->suspendFire();
    this->observed->SetAutoFocusOffset(offset);
    this->resumeFire();
    this->fireChanged(DIRTY_AUTOFOCUSOFFSET, false);
}


/*
 * vislib::graphics::ObservableCameraParams::SetClip
 */
void vislib::graphics::ObservableCameraParams::SetClip(
        SceneSpaceType nearClip, SceneSpaceType farClip) {
    this->suspendFire();
    this->observed->SetClip(nearClip, farClip);
    this->resumeFire();
    this->fireChanged(DIRTY_NEARCLIP | DIRTY_FARCLIP);

}


/*
 * vislib::graphics::ObservableCameraParams::SetCoordSystemType
 */
void vislib::graphics::ObservableCameraParams::SetCoordSystemType(
        vislib::math::CoordSystemType coordSysType) {
    this->suspendFire();
    this->observed->SetCoordSystemType(coordSysType);
    this->resumeFire();
    this->fireChanged(DIRTY_COORDSYSTEMTYPE);
}


/*
 * vislib::graphics::ObservableCameraParams::SetEye
 */
void vislib::graphics::ObservableCameraParams::SetEye(StereoEye eye) {
    this->suspendFire();
    this->observed->SetEye(eye);
    this->resumeFire();
    this->fireChanged(DIRTY_EYE, false);
}


/*
 * vislib::graphics::ObservableCameraParams::SetFarClip
 */
void vislib::graphics::ObservableCameraParams::SetFarClip(
        SceneSpaceType farClip) {
    this->suspendFire();
    this->observed->SetFarClip(farClip);
    this->resumeFire();
    this->fireChanged(DIRTY_NEARCLIP | DIRTY_FARCLIP);
}


/*
 * vislib::graphics::ObservableCameraParams::SetFocalDistance
 */
void vislib::graphics::ObservableCameraParams::SetFocalDistance(
        SceneSpaceType focalDistance) {
    this->suspendFire();
    this->observed->SetFocalDistance(focalDistance);
    this->resumeFire();
    this->fireChanged(DIRTY_FOCALDISTANCE, false);
}


/*
 * vislib::graphics::ObservableCameraParams::SetLimits
 */
void vislib::graphics::ObservableCameraParams::SetLimits(
        const SmartPtr<CameraParameterLimits>& limits) {
    this->observed->SetLimits(limits);
}


/*
 * vislib::graphics::ObservableCameraParams::SetLookAt
 */
void vislib::graphics::ObservableCameraParams::SetLookAt(
        const math::Point<SceneSpaceType, 3>& lookAt) {
    this->suspendFire();
    this->observed->SetLookAt(lookAt);
    this->resumeFire();
    this->fireChanged(DIRTY_LOOKAT);
}


/*
 * vislib::graphics::ObservableCameraParams::SetNearClip
 */
void vislib::graphics::ObservableCameraParams::SetNearClip(
        SceneSpaceType nearClip) {
    this->suspendFire();
    this->observed->SetNearClip(nearClip);
    this->resumeFire();
    this->fireChanged(DIRTY_NEARCLIP | DIRTY_FARCLIP);
}


/*
 * vislib::graphics::ObservableCameraParams::SetPosition
 */
void vislib::graphics::ObservableCameraParams::SetPosition(
        const math::Point<SceneSpaceType, 3>& position) {
    this->suspendFire();
    this->observed->SetPosition(position);
    this->resumeFire();
    this->fireChanged(DIRTY_POSITION);
}


/*
 * vislib::graphics::ObservableCameraParams::SetProjection
 */
void vislib::graphics::ObservableCameraParams::SetProjection(
        ProjectionType projectionType) {
    this->suspendFire();
    this->observed->SetProjection(projectionType);
    this->resumeFire();
    this->fireChanged(DIRTY_PROJECTION);
}


/*
 * vislib::graphics::ObservableCameraParams::SetStereoDisparity
 */
void vislib::graphics::ObservableCameraParams::SetStereoDisparity(
        SceneSpaceType stereoDisparity) {
    this->suspendFire();
    this->observed->SetStereoDisparity(stereoDisparity);
    this->resumeFire();
    this->fireChanged(DIRTY_DISPARITY);
}


/*
 * vislib::graphics::ObservableCameraParams::SetStereoParameters
 */
void vislib::graphics::ObservableCameraParams::SetStereoParameters(
        SceneSpaceType stereoDisparity, StereoEye eye, 
        SceneSpaceType focalDistance) {
    this->suspendFire();
    this->observed->SetStereoParameters(stereoDisparity, eye, focalDistance);
    this->resumeFire();
    this->fireChanged(DIRTY_DISPARITY | DIRTY_EYE | DIRTY_FOCALDISTANCE);
}


/*
 * vislib::graphics::ObservableCameraParams::SetTileRect
 */
void vislib::graphics::ObservableCameraParams::SetTileRect(
        const math::Rectangle<ImageSpaceType>& tileRect) {
    this->suspendFire();
    this->observed->SetTileRect(tileRect);
    this->resumeFire();
    this->fireChanged(DIRTY_TILERECT, false);
}


/*
 * vislib::graphics::ObservableCameraParams::SetUp
 */
void vislib::graphics::ObservableCameraParams::SetUp(
        const math::Vector<SceneSpaceType, 3>& up) {
    this->suspendFire();
    this->observed->SetUp(up);
    this->resumeFire();
    this->fireChanged(DIRTY_UP);
}


/*
 * vislib::graphics::ObservableCameraParams::SetView
 */
void vislib::graphics::ObservableCameraParams::SetView(
        const math::Point<SceneSpaceType, 3>& position, 
        const math::Point<SceneSpaceType, 3>& lookAt, 
        const math::Vector<SceneSpaceType, 3>& up) {
    this->suspendFire();
    this->observed->SetView(position, lookAt, up);
    this->resumeFire();
    this->fireChanged(DIRTY_POSITION | DIRTY_LOOKAT | DIRTY_UP);
}


/* 
 * vislib::graphics::ObservableCameraParams::SetVirtualViewSize
 */
void vislib::graphics::ObservableCameraParams::SetVirtualViewSize(
        const math::Dimension<ImageSpaceType, 2>& viewSize) {
    this->suspendFire();
    this->observed->SetVirtualViewSize(viewSize);
    this->resumeFire();
    this->fireChanged(DIRTY_VIRTUALVIEW);
}


/*
 * vislib::graphics::ObservableCameraParams::SyncNumber
 */
unsigned int vislib::graphics::ObservableCameraParams::SyncNumber(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->SyncNumber();
}


/*
 * vislib::graphics::ObservableCameraParams::TileRect
 */
const vislib::math::Rectangle<vislib::graphics::ImageSpaceType>& 
vislib::graphics::ObservableCameraParams::TileRect(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->TileRect();
}


/*
 * vislib::graphics::ObservableCameraParams::Up
 */
const vislib::math::Vector<vislib::graphics::SceneSpaceType, 3>& 
vislib::graphics::ObservableCameraParams::Up(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->Up();
}


/*
 * vislib::graphics::ObservableCameraParams::VirtualViewSize
 */
const vislib::math::Dimension<vislib::graphics::ImageSpaceType, 2>& 
vislib::graphics::ObservableCameraParams::VirtualViewSize(void) const {
    ASSERT(!this->observed.IsNull());
    return this->observed->VirtualViewSize();
}


/*
 * vislib::graphics::ObservableCameraParams::operator =
 */
vislib::graphics::ObservableCameraParams& 
vislib::graphics::ObservableCameraParams::operator =(
        const ObservableCameraParams& rhs) {
    if (this != &rhs)  {
        this->suspendFire();
        this->observed->operator =(rhs);
        this->resumeFire();
        this->fireChanged();
    }

    return *this;
}


/*
 * vislib::graphics::ObservableCameraParams::operator ==
 */
bool vislib::graphics::ObservableCameraParams::operator ==(
        const ObservableCameraParams& rhs) const {
    return this->observed->operator ==(rhs);
}


/*
 * vislib::graphics::ObservableCameraParams::DIRTY_APERTUREANGLE 
 */
const UINT32 vislib::graphics::ObservableCameraParams::DIRTY_ALL 
    = 0xFFFFFFFF;


/*
 * vislib::graphics::ObservableCameraParams::DIRTY_APERTUREANGLE 
 */
const UINT32 vislib::graphics::ObservableCameraParams::DIRTY_APERTUREANGLE
    = 0x00000001;


/*
 * vislib::graphics::ObservableCameraParams::DIRTY_APERTUREANGLE 
 */
const UINT32 vislib::graphics::ObservableCameraParams::DIRTY_AUTOFOCUSOFFSET
    = 0x00004000;


/*
 * vislib::graphics::ObservableCameraParams::DIRTY_COORDSYSTEMTYPE
 */
const UINT32 vislib::graphics::ObservableCameraParams::DIRTY_COORDSYSTEMTYPE
    = 0x00002000;


/*
 * vislib::graphics::ObservableCameraParams::DIRTY_EYE
 */
const UINT32 vislib::graphics::ObservableCameraParams::DIRTY_EYE
    = 0x00000002;


/*
 * vislib::graphics::ObservableCameraParams::DIRTY_FARCLIP
 */
const UINT32 vislib::graphics::ObservableCameraParams::DIRTY_FARCLIP
    = 0x00000004;


/*
 * vislib::graphics::ObservableCameraParams::DIRTY_FOCALDISTANCE
 */
const UINT32 vislib::graphics::ObservableCameraParams::DIRTY_FOCALDISTANCE
    = 0x00000008;


/*
 * vislib::graphics::ObservableCameraParams::DIRTY_LIMITS
 */
const UINT32 vislib::graphics::ObservableCameraParams::DIRTY_LIMITS
    = 0x00000010;


/*
 * vislib::graphics::ObservableCameraParams::DIRTY_LOOKAT
 */
const UINT32 vislib::graphics::ObservableCameraParams::DIRTY_LOOKAT
    = 0x00000020;


/*
 * vislib::graphics::ObservableCameraParams::DIRTY_NEARCLIP
 */
const UINT32 vislib::graphics::ObservableCameraParams::DIRTY_NEARCLIP
    = 0x00000040;


/*
 * vislib::graphics::ObservableCameraParams::DIRTY_POSITION
 */
const UINT32 vislib::graphics::ObservableCameraParams::DIRTY_POSITION
    = 0x00000080;


/*
 * vislib::graphics::ObservableCameraParams::DIRTY_PROJECTION 
 */
const UINT32 vislib::graphics::ObservableCameraParams::DIRTY_PROJECTION 
    = 0x00000100;


/*
 * vislib::graphics::ObservableCameraParams::DIRTY_DISPARITY
 */
const UINT32 vislib::graphics::ObservableCameraParams::DIRTY_DISPARITY
    = 0x00000200;


/*
 * vislib::graphics::ObservableCameraParams::DIRTY_TILERECT
 */
const UINT32 vislib::graphics::ObservableCameraParams::DIRTY_TILERECT
    = 0x00000400;


/*
 * vislib::graphics::ObservableCameraParams::DIRTY_UP
 */
const UINT32 vislib::graphics::ObservableCameraParams::DIRTY_UP
    = 0x00000800;
        

/*
 * vislib::graphics::ObservableCameraParams::DIRTY_VIRTUALVIEW
 */
const UINT32 vislib::graphics::ObservableCameraParams::DIRTY_VIRTUALVIEW
    = 0x00001000;


/*
 * vislib::graphics::ObservableCameraParams::fireChanged
 */
void vislib::graphics::ObservableCameraParams::fireChanged(
        const UINT32 which, const bool andAllDirty) {

    if (this->isBatchInteraction || this->isSuspendFire) {
        /* Firing events is suspended, just mark fields dirty. */
        this->dirtyFields |= which;

    } else {
        /* Fire events. */
        SingleLinkedList<CameraParameterObserver *>::Iterator it 
            = const_cast<ObservableCameraParams *>(this)
            ->camParamObservers.GetIterator();

        while (it.HasNext()) {
#define IMPLEMENT_FIRE_EX(name, flag, value)                                   \
            if (((which & (flag)) != 0)                                        \
                    || (andAllDirty && ((this->dirtyFields & (flag)) != 0))) { \
                VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Firing On" #name  \
                    "Changed ...\n");                                          \
                observer->On##name##Changed(value);                            \
            } 
#define IMPLEMENT_FIRE(name, flag) IMPLEMENT_FIRE_EX(name, flag, this->name())

            CameraParameterObserver *observer = it.Next();

            IMPLEMENT_FIRE(ApertureAngle, DIRTY_APERTUREANGLE);
            //IMPLEMENT_FIRE(AutoFocusOffset, DIRTY_AUTOFOCUSOFFSET);
            IMPLEMENT_FIRE(Eye, DIRTY_EYE);
            IMPLEMENT_FIRE(FarClip, DIRTY_FARCLIP);
            IMPLEMENT_FIRE(FocalDistance, DIRTY_FOCALDISTANCE);
            //IMPLEMENT_FIRE_EX(OnLimitsChanged, DIRTY_LIMITS, this->Limits());
            IMPLEMENT_FIRE(LookAt, DIRTY_LOOKAT);
            IMPLEMENT_FIRE(NearClip, DIRTY_NEARCLIP);
            IMPLEMENT_FIRE(Position, DIRTY_POSITION);
            IMPLEMENT_FIRE(Projection, DIRTY_PROJECTION);
            IMPLEMENT_FIRE(StereoDisparity, DIRTY_DISPARITY);
            IMPLEMENT_FIRE(TileRect, DIRTY_TILERECT);
            IMPLEMENT_FIRE(Up, DIRTY_UP);
            IMPLEMENT_FIRE(VirtualViewSize, DIRTY_VIRTUALVIEW);

#undef IMPLEMENT_FIRE
#undef IMPLEMENT_FIRE_EX
        }
    } /* end if (this->isBatchInteraction || this->isSuspendFire) */
}


/*
 * vislib::graphics::ObservableCameraParams::preBaseSet
 */
void vislib::graphics::ObservableCameraParams::preBaseSet(
        const SmartPtr<CameraParameters>& params) {
    // Don't care.
}


/*
 * vislib::graphics::ObservableCameraParams::resetOverride
 */
void vislib::graphics::ObservableCameraParams::resetOverride(void) {
    // We do not override anything.
}
