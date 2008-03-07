/*
 * CameraParameterObserver.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/CameraParameterObserver.h"


/*
 * vislib::graphics::CameraParameterObserver::~CameraParameterObserver
 */
vislib::graphics::CameraParameterObserver::~CameraParameterObserver(void) {
    // Nothing to do.
}


/* 
 * vislib::graphics::CameraParameterObserver::OnApertureAngleChanged
 */
void vislib::graphics::CameraParameterObserver::OnApertureAngleChanged(
        const math::AngleDeg newValue) {
    // Nothing to do.
}


/* 
 * vislib::graphics::CameraParameterObserver::OnEyeChanged
 */
void vislib::graphics::CameraParameterObserver::OnEyeChanged(
       const CameraParameters::StereoEye newValue) {
    // Nothing to do.
}


/* 
 * vislib::graphics::CameraParameterObserver::OnFarClipChanged
 */
void vislib::graphics::CameraParameterObserver::OnFarClipChanged(
        const SceneSpaceType newValue) {
    // Nothing to do.
}


/* 
 * vislib::graphics::CameraParameterObserver::OnFocalDistanceChanged
 */
void vislib::graphics::CameraParameterObserver::OnFocalDistanceChanged(
        const SceneSpaceType newValue) {
    // Nothing to do.
}


/* 
 * vislib::graphics::CameraParameterObserver::OnLookAtChanged
 */
void vislib::graphics::CameraParameterObserver::OnLookAtChanged(
        const math::Point<SceneSpaceType, 3>& newValue) {
    // Nothing to do.
}


/* 
 * vislib::graphics::CameraParameterObserver::OnNearClipChanged
 */
void vislib::graphics::CameraParameterObserver::OnNearClipChanged(
        const SceneSpaceType newValue) {
    // Nothing to do.
}


/* 
 * vislib::graphics::CameraParameterObserver::OnPositionChanged
 */
void vislib::graphics::CameraParameterObserver::OnPositionChanged(
        const math::Point<SceneSpaceType, 3>& newValue) {
    // Nothing to do.
}


/* 
 * vislib::graphics::CameraParameterObserver::OnProjectionChanged
 */
void vislib::graphics::CameraParameterObserver::OnProjectionChanged(
        const CameraParameters::ProjectionType newValue) {
    // Nothing to do.
}


/* 
 * vislib::graphics::CameraParameterObserver::OnStereoDisparityChanged
 */
void vislib::graphics::CameraParameterObserver::OnStereoDisparityChanged(
        const SceneSpaceType newValue) {
    // Nothing to do.
}


/* 
 * vislib::graphics::CameraParameterObserver::OnTileRectChanged
 */
void vislib::graphics::CameraParameterObserver::OnTileRectChanged(
        const math::Rectangle<ImageSpaceType>& newValue) {
    // Nothing to do.
}


/* 
 * vislib::graphics::CameraParameterObserver::OnUpChanged
 */
void vislib::graphics::CameraParameterObserver::OnUpChanged(
        const math::Vector<SceneSpaceType, 3>& newValue) {
    // Nothing to do.
}


/* 
 * vislib::graphics::CameraParameterObserver::OnVirtualViewSizeChanged
 */
void vislib::graphics::CameraParameterObserver::OnVirtualViewSizeChanged(
       const math::Dimension<ImageSpaceType, 2>& newValue) {
    // Nothing to do.
}


/*
 * vislib::graphics::CameraParameterObserver::CameraParameterObserver
 */
vislib::graphics::CameraParameterObserver::CameraParameterObserver(void) {
    // Nothing to do.
}