/*
 * AbstractControllerNode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractControllerNode.h"

#include "vislib/clustermessages.h"


/*
 * vislib::net::cluster::AbstractControllerNode::~AbstractControllerNode
 */
vislib::net::cluster::AbstractControllerNode::~AbstractControllerNode(void) {
}


/*
 * vislib::net::cluster::AbstractControllerNode::BeginBatchInteraction
 */
void vislib::net::cluster::AbstractControllerNode::BeginBatchInteraction(void) {
}


/*
 * vislib::net::cluster::AbstractControllerNode::EndBatchInteraction
 */
void vislib::net::cluster::AbstractControllerNode::EndBatchInteraction(void) {
}



/* 
 * vislib::net::cluster::AbstractControllerNode::OnApertureAngleChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnApertureAngleChanged(
        const math::AngleDeg newValue) {
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnEyeChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnEyeChanged(
       const graphics::CameraParameters::StereoEye newValue) {
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnFarClipChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnFarClipChanged(
        const graphics::SceneSpaceType newValue) {
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnFocalDistanceChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnFocalDistanceChanged(
        const graphics::SceneSpaceType newValue) {
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnLookAtChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnLookAtChanged(
        const graphics::SceneSpacePoint3D& newValue) {
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnNearClipChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnNearClipChanged(
        const graphics::SceneSpaceType newValue) {
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnPositionChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnPositionChanged(
        const graphics::SceneSpacePoint3D& newValue) {
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnProjectionChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnProjectionChanged(
        const graphics::CameraParameters::ProjectionType newValue) {
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnStereoDisparityChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnStereoDisparityChanged(
        const graphics::SceneSpaceType newValue) {
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnTileRectChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnTileRectChanged(
        const graphics::ImageSpaceRectangle& newValue) {
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnUpChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnUpChanged(
        const graphics::SceneSpaceVector3D& newValue) {
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnVirtualViewSizeChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnVirtualViewSizeChanged(
        const graphics::ImageSpaceDimension& newValue) {
}



/*
 * vislib::net::cluster::AbstractControllerNode::AbstractControllerNode
 */
vislib::net::cluster::AbstractControllerNode::AbstractControllerNode(void) {
}

