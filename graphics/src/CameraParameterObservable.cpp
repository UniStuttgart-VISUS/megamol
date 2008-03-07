/*
 * CameraParameterObservable.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/CameraParameterObservable.h"

#include "vislib/assert.h"


/*
 * vislib::graphics::CameraParameterObservable::~CameraParameterObservable
 */
vislib::graphics::CameraParameterObservable::~CameraParameterObservable(void) {
}


/*
 * vislib::graphics::CameraParameterObservable::AddCameraParameterObserver
 */
void vislib::graphics::CameraParameterObservable::AddCameraParameterObserver(
        CameraParameterObserver *observer) {
    ASSERT(observer != NULL);

    if ((observer != NULL) && !this->camParamObservers.Contains(observer)) {
        this->camParamObservers.Append(observer);
    }
}


/*
 * vislib::graphics::CameraParameterObservable::RemoveCameraParameterObserver
 */
void vislib::graphics::CameraParameterObservable::RemoveCameraParameterObserver(
        CameraParameterObserver *observer) {
    ASSERT(observer != NULL);
    this->camParamObservers.RemoveAll(observer);
}


/*
 * vislib::graphics::CameraParameterObservable::CameraParameterObservable
 */
vislib::graphics::CameraParameterObservable::CameraParameterObservable(void) {
    // Nothing to to.
}


#define IMPLEMENT_FIRE_CHANGED(attribType, eventName)                          \
void vislib::graphics::CameraParameterObservable::fire##eventName(             \
       const attribType newValue) {                                            \
    SingleLinkedList<CameraParameterObserver *>::Iterator it =                 \
        const_cast<CameraParameterObservable *>(this)                          \
        ->camParamObservers.GetIterator();                                     \
    while (it.HasNext()) {                                                     \
        it.Next()->On##eventName(newValue);                                    \
    }                                                                          \
}

IMPLEMENT_FIRE_CHANGED(math::AngleDeg, ApertureAngleChanged);
IMPLEMENT_FIRE_CHANGED(CameraParameters::StereoEye, EyeChanged);
IMPLEMENT_FIRE_CHANGED(SceneSpaceType, FarClipChanged);
IMPLEMENT_FIRE_CHANGED(SceneSpaceType, FocalDistanceChanged);
IMPLEMENT_FIRE_CHANGED(SceneSpacePoint3D&, LookAtChanged);
IMPLEMENT_FIRE_CHANGED(SceneSpaceType, NearClipChanged);
IMPLEMENT_FIRE_CHANGED(SceneSpacePoint3D&, PositionChanged);
IMPLEMENT_FIRE_CHANGED(CameraParameters::ProjectionType, ProjectionChanged);
IMPLEMENT_FIRE_CHANGED(SceneSpaceType, StereoDisparityChanged);
IMPLEMENT_FIRE_CHANGED(ImageSpaceRectangle&, TileRectChanged);
IMPLEMENT_FIRE_CHANGED(SceneSpaceVector3D&, UpChanged);
IMPLEMENT_FIRE_CHANGED(ImageSpaceDimension&, VirtualViewSizeChanged);

#undef IMPLEMENT_FIRE_CHANGED