/*
 * AbstractControllerNode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractControllerNode.h"


/*
 * vislib::net::cluster::AbstractControllerNode::~AbstractControllerNode
 */
vislib::net::cluster::AbstractControllerNode::~AbstractControllerNode(void) {
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnApertureAngleChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnApertureAngleChanged(
        const math::AngleDeg newValue) {
    this->sendIntegralCamParam(MSGID_APERTUREANGLE, newValue);
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnEyeChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnEyeChanged(
       const graphics::CameraParameters::StereoEye newValue) {
    this->sendIntegralCamParam(MSGID_EYE, newValue);
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnFarClipChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnFarClipChanged(
        const graphics::SceneSpaceType newValue) {
    this->sendIntegralCamParam(MSGID_FARCLIP, newValue);
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnFocalDistanceChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnFocalDistanceChanged(
        const graphics::SceneSpaceType newValue) {
    this->sendIntegralCamParam(MSGID_FOCALDISTANCE, newValue);
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnLookAtChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnLookAtChanged(
        const graphics::SceneSpacePoint3D& newValue) {
    this->sendVectorialCamParam<graphics::SceneSpaceType, 3>(MSGID_LOOKAT, 
        newValue.PeekCoordinates());
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnNearClipChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnNearClipChanged(
        const graphics::SceneSpaceType newValue) {
    this->sendIntegralCamParam(MSGID_NEARCLIP, newValue);
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnPositionChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnPositionChanged(
        const graphics::SceneSpacePoint3D& newValue) {
    this->sendVectorialCamParam<graphics::SceneSpaceType, 3>(MSGID_POSITION,
        newValue.PeekCoordinates());
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnProjectionChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnProjectionChanged(
        const graphics::CameraParameters::ProjectionType newValue) {
    this->sendIntegralCamParam(MSGID_PROJECTION, newValue);
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnStereoDisparityChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnStereoDisparityChanged(
        const graphics::SceneSpaceType newValue) {
    this->sendIntegralCamParam(MSGID_STEREODISPARITY, newValue);
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnTileRectChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnTileRectChanged(
        const graphics::ImageSpaceRectangle& newValue) {
    this->sendVectorialCamParam<graphics::ImageSpaceType, 4>(MSGID_TILERECT,
        newValue.PeekBounds());
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnUpChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnUpChanged(
        const graphics::SceneSpaceVector3D& newValue) {
    this->sendVectorialCamParam<graphics::SceneSpaceType, 3>(MSGID_UP,
        newValue.PeekComponents());
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnVirtualViewSizeChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnVirtualViewSizeChanged(
        const graphics::ImageSpaceDimension& newValue) {
    this->sendVectorialCamParam<graphics::ImageSpaceType, 2>(
        MSGID_VIRTUALVIEWSIZE, newValue.PeekDimension());
}


/*
 * vislib::net::cluster::AbstractControllerNode::AbstractControllerNode
 */
vislib::net::cluster::AbstractControllerNode::AbstractControllerNode(void) 
        : AbstractClusterNode(), graphics::CameraParameterObserver() {
    // Nothing to do.
}


/*
 * vislib::net::cluster::AbstractControllerNode::AbstractControllerNode
 */
vislib::net::cluster::AbstractControllerNode::AbstractControllerNode(
        const AbstractControllerNode& rhs) 
        : AbstractClusterNode(rhs), graphics::CameraParameterObserver(rhs) {
    // Nothing to do.
}


/*
 * vislib::net::cluster::AbstractControllerNode::operator =
 */
vislib::net::cluster::AbstractControllerNode& 
vislib::net::cluster::AbstractControllerNode::operator =(
        const AbstractControllerNode& rhs) {
    AbstractClusterNode::operator =(rhs);
    graphics::CameraParameterObserver::operator =(rhs);
    return *this;
}
