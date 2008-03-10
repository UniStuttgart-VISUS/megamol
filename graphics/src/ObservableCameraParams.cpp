/*
 * ObservableCameraParams.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/ObservableCameraParams.h"

#include "vislib/assert.h"


/*
 * vislib::graphics::ObservableCameraParams::ObservableCameraParams
 */
vislib::graphics::ObservableCameraParams::ObservableCameraParams(
        SmartPtr<CameraParameters>& base) 
        : Super(base), dirtyFields(0), isBatchInteraction(false), 
        isSuspendFire(false) {
    // No one can have registered here, so it is unnecessary to fire the event.
}


/*
 * vislib::graphics::ObservableCameraParams::ObservableCameraParams
 */
vislib::graphics::ObservableCameraParams::ObservableCameraParams(
        const ObservableCameraParams& rhs) 
        : Super(rhs), dirtyFields(0), isBatchInteraction(false), 
        isSuspendFire(false) {
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
    Super::ApplyLimits();
    this->resumeFire();
    this->fireChanged();
}


/*
 * vislib::graphics::ObservableCameraParams::BeginBatchInteraction
 */
void vislib::graphics::ObservableCameraParams::BeginBatchInteraction(void) {
    this->isBatchInteraction = true;
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
    Super::Reset();
    this->resumeFire();
    this->fireChanged();
}


/*
 * vislib::graphics::ObservableCameraParams::ResetTileRect
 */
void vislib::graphics::ObservableCameraParams::ResetTileRect(void) {
    this->suspendFire();
    Super::ResetTileRect();
    this->resumeFire();
    this->fireChanged(DIRTY_TILERECT, false);
}


/*
 * vislib::graphics::ObservableCameraParams::SetApertureAngle
 */
void vislib::graphics::ObservableCameraParams::SetApertureAngle(
        math::AngleDeg apertureAngle) {
    this->suspendFire();
    Super::SetApertureAngle(apertureAngle);
    this->resumeFire();
    this->fireChanged(DIRTY_APERTUREANGLE, false);
}


/*
 * vislib::graphics::ObservableCameraParams::SetClip
 */
void vislib::graphics::ObservableCameraParams::SetClip(
        SceneSpaceType nearClip, SceneSpaceType farClip) {
    this->suspendFire();
    Super::SetClip(nearClip, farClip);
    this->resumeFire();
    this->fireChanged(DIRTY_NEARCLIP | DIRTY_FARCLIP);

}


/*
 * vislib::graphics::ObservableCameraParams::SetEye
 */
void vislib::graphics::ObservableCameraParams::SetEye(StereoEye eye) {
    this->suspendFire();
    Super::SetEye(eye);
    this->resumeFire();
    this->fireChanged(DIRTY_EYE, false);
}


/*
 * vislib::graphics::ObservableCameraParams::SetFarClip
 */
void vislib::graphics::ObservableCameraParams::SetFarClip(
        SceneSpaceType farClip) {
    this->suspendFire();
    Super::SetFarClip(farClip);
    this->resumeFire();
    this->fireChanged(DIRTY_NEARCLIP | DIRTY_FARCLIP);
}


/*
 * vislib::graphics::ObservableCameraParams::SetFocalDistance
 */
void vislib::graphics::ObservableCameraParams::SetFocalDistance(
        SceneSpaceType focalDistance) {
    this->suspendFire();
    Super::SetFocalDistance(focalDistance);
    this->resumeFire();
    this->fireChanged(DIRTY_FOCALDISTANCE, false);
}


/*
 * vislib::graphics::ObservableCameraParams::SetLookAt
 */
void vislib::graphics::ObservableCameraParams::SetLookAt(
        const math::Point<SceneSpaceType, 3>& lookAt) {
    this->suspendFire();
    Super::SetLookAt(lookAt);
    this->resumeFire();
    this->fireChanged(DIRTY_LOOKAT);
}


/*
 * vislib::graphics::ObservableCameraParams::SetNearClip
 */
void vislib::graphics::ObservableCameraParams::SetNearClip(
        SceneSpaceType nearClip) {
    this->suspendFire();
    Super::SetNearClip(nearClip);
    this->resumeFire();
    this->fireChanged(DIRTY_NEARCLIP | DIRTY_FARCLIP);
}


/*
 * vislib::graphics::ObservableCameraParams::SetPosition
 */
void vislib::graphics::ObservableCameraParams::SetPosition(
        const math::Point<SceneSpaceType, 3>& position) {
    this->suspendFire();
    Super::SetPosition(position);
    this->resumeFire();
    this->fireChanged(DIRTY_POSITION);
}


/*
 * vislib::graphics::ObservableCameraParams::SetProjection
 */
void vislib::graphics::ObservableCameraParams::SetProjection(
        ProjectionType projectionType) {
    this->suspendFire();
    Super::SetProjection(projectionType);
    this->resumeFire();
    this->fireChanged(DIRTY_PROJECTION);
}


/*
 * vislib::graphics::ObservableCameraParams::SetStereoDisparity
 */
void vislib::graphics::ObservableCameraParams::SetStereoDisparity(
        SceneSpaceType stereoDisparity) {
    this->suspendFire();
    Super::SetStereoDisparity(stereoDisparity);
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
    Super::SetStereoParameters(stereoDisparity, eye, focalDistance);
    this->resumeFire();
    this->fireChanged(DIRTY_DISPARITY | DIRTY_EYE | DIRTY_FOCALDISTANCE);
}


/*
 * vislib::graphics::ObservableCameraParams::SetTileRect
 */
void vislib::graphics::ObservableCameraParams::SetTileRect(
        const math::Rectangle<ImageSpaceType>& tileRect) {
    this->suspendFire();
    Super::SetTileRect(tileRect);
    this->resumeFire();
    this->fireChanged(DIRTY_TILERECT, false);
}


/*
 * vislib::graphics::ObservableCameraParams::SetUp
 */
void vislib::graphics::ObservableCameraParams::SetUp(
        const math::Vector<SceneSpaceType, 3>& up) {
    this->suspendFire();
    Super::SetUp(up);
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
    Super::SetView(position, lookAt, up);
    this->resumeFire();
    this->fireChanged(DIRTY_POSITION | DIRTY_LOOKAT | DIRTY_UP);
}


/* 
 * vislib::graphics::ObservableCameraParams::SetVirtualViewSize
 */
void vislib::graphics::ObservableCameraParams::SetVirtualViewSize(
        const math::Dimension<ImageSpaceType, 2>& viewSize) {
    this->suspendFire();
    Super::SetVirtualViewSize(viewSize);
    this->resumeFire();
    this->fireChanged(DIRTY_VIRTUALVIEW);
}


/*
 * vislib::graphics::ObservableCameraParams::operator =
 */
vislib::graphics::ObservableCameraParams& 
vislib::graphics::ObservableCameraParams::operator =(
        const ObservableCameraParams& rhs) {
    if (this != &rhs)  {
        this->suspendFire();
        Super::operator =(rhs);
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
    return Super::operator ==(rhs);
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
                observer->On##name##Changed(value);                            \
            } 
#define IMPLEMENT_FIRE(name, flag) IMPLEMENT_FIRE_EX(name, flag, this->name())

            CameraParameterObserver *observer = it.Next();

            IMPLEMENT_FIRE(ApertureAngle, DIRTY_APERTUREANGLE);
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
