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
    this->isBatchInteraction = false;
}


/*
 * vislib::net::cluster::AbstractControllerNode::EndBatchInteraction
 */
void vislib::net::cluster::AbstractControllerNode::EndBatchInteraction(void) {
    if (this->isBatchInteraction) {
        // TODO: Transfer now
        this->isBatchInteraction = false;
    }
}



/* 
 * vislib::net::cluster::AbstractControllerNode::OnApertureAngleChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnApertureAngleChanged(
        const math::AngleDeg newValue) {
    if (this->isBatchInteraction) {
        this->batchDirtyFields |= DIRTY_APERTUREANGLE;
    } else {
        // TODO: Transfer now
    }
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnEyeChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnEyeChanged(
       const graphics::CameraParameters::StereoEye newValue) {
    if (this->isBatchInteraction) {
        this->batchDirtyFields |= DIRTY_EYE;
    } else {
        // TODO: Transfer now
    }
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnFarClipChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnFarClipChanged(
        const graphics::SceneSpaceType newValue) {
    if (this->isBatchInteraction) {
        this->batchDirtyFields |= DIRTY_FARCLIP;
    } else {
        // TODO: Transfer now
    }
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnFocalDistanceChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnFocalDistanceChanged(
        const graphics::SceneSpaceType newValue) {
    if (this->isBatchInteraction) {
        this->batchDirtyFields |= DIRTY_FOCALDISTANCE;
    } else {
        // TODO: Transfer now
    }
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnLookAtChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnLookAtChanged(
        const graphics::SceneSpacePoint3D& newValue) {
    if (this->isBatchInteraction) {
        this->batchDirtyFields |= DIRTY_LOOKAT;
    } else {
        // TODO: Transfer now
    }
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnNearClipChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnNearClipChanged(
        const graphics::SceneSpaceType newValue) {
    if (this->isBatchInteraction) {
        this->batchDirtyFields |= DIRTY_NEARCLIP;
    } else {
        // TODO: Transfer now
    }
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnPositionChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnPositionChanged(
        const graphics::SceneSpacePoint3D& newValue) {
    if (this->isBatchInteraction) {
        this->batchDirtyFields |= DIRTY_POSITION;
    } else {
        // TODO: Transfer now
    }
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnProjectionChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnProjectionChanged(
        const graphics::CameraParameters::ProjectionType newValue) {
    if (this->isBatchInteraction) {
        this->batchDirtyFields |= DIRTY_PROJECTION;
    } else {
        // TODO: Transfer now
    }
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnStereoDisparityChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnStereoDisparityChanged(
        const graphics::SceneSpaceType newValue) {
    if (this->isBatchInteraction) {
        this->batchDirtyFields |= DIRTY_STEREODISPARITY;
    } else {
        // TODO: Transfer now
    }
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnTileRectChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnTileRectChanged(
        const graphics::ImageSpaceRectangle& newValue) {
    if (this->isBatchInteraction) {
        this->batchDirtyFields |= DIRTY_TILERECT;
    } else {
        // TODO: Transfer now
    }
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnUpChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnUpChanged(
        const graphics::SceneSpaceVector3D& newValue) {
    if (this->isBatchInteraction) {
        this->batchDirtyFields |= DIRTY_UP;
    } else {
        // TODO: Transfer now
    }
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnVirtualViewSizeChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnVirtualViewSizeChanged(
        const graphics::ImageSpaceDimension& newValue) {
    if (this->isBatchInteraction) {
        this->batchDirtyFields |= DIRTY_VIRTUALVIEWSIZE;
    } else {
        // TODO: Transfer now
    }
}



/*
 * vislib::net::cluster::AbstractControllerNode::AbstractControllerNode
 */
vislib::net::cluster::AbstractControllerNode::AbstractControllerNode(void) 
        : batchDirtyFields(0), isBatchInteraction(false) {
}


/*
 * vislib::net::cluster::AbstractControllerNode::DIRTY_APERTUREANGLE 
 */
const UINT32 vislib::net::cluster::AbstractControllerNode::DIRTY_APERTUREANGLE 
    = 0x00000001;


/*
 * vislib::net::cluster::AbstractControllerNode::DIRTY_EYE
 */
const UINT32 vislib::net::cluster::AbstractControllerNode::DIRTY_EYE
    = 0x00000002;


/*
 * vislib::net::cluster::AbstractControllerNode::DIRTY_FARCLIP
 */
const UINT32 vislib::net::cluster::AbstractControllerNode::DIRTY_FARCLIP
    = 0x00000004;


/*
 * vislib::net::cluster::AbstractControllerNode::DIRTY_FOCALDISTANCE
 */
const UINT32 vislib::net::cluster::AbstractControllerNode::DIRTY_FOCALDISTANCE
    = 0x00000008;


/*
 * vislib::net::cluster::AbstractControllerNode::DIRTY_LIMITS
 */
const UINT32 vislib::net::cluster::AbstractControllerNode::DIRTY_LIMITS
    = 0x00000010;


/*
 * vislib::net::cluster::AbstractControllerNode::DIRTY_LOOKAT
 */
const UINT32 vislib::net::cluster::AbstractControllerNode::DIRTY_LOOKAT
    = 0x00000020;


/*
 * vislib::net::cluster::AbstractControllerNode::DIRTY_NEARCLIP
 */
const UINT32 vislib::net::cluster::AbstractControllerNode::DIRTY_NEARCLIP
    = 0x00000040;


/*
 * vislib::net::cluster::AbstractControllerNode::DIRTY_POSITION
 */
const UINT32 vislib::net::cluster::AbstractControllerNode::DIRTY_POSITION
    = 0x00000080;


/*
 * vislib::net::cluster::AbstractControllerNode::DIRTY_PROJECTION 
 */
const UINT32 vislib::net::cluster::AbstractControllerNode::DIRTY_PROJECTION 
    = 0x00000100;


/*
 * vislib::net::cluster::AbstractControllerNode::DIRTY_STEREODISPARITY
 */
const UINT32 vislib::net::cluster::AbstractControllerNode::DIRTY_STEREODISPARITY
    = 0x00000200;


/*
 * vislib::net::cluster::AbstractControllerNode::DIRTY_TILERECT
 */
const UINT32 vislib::net::cluster::AbstractControllerNode::DIRTY_TILERECT
    = 0x00000400;


/*
 * vislib::net::cluster::AbstractControllerNode::DIRTY_UP
 */
const UINT32 vislib::net::cluster::AbstractControllerNode::DIRTY_UP
    = 0x00000800;
        

/*
 * vislib::net::cluster::AbstractControllerNode::DIRTY_VIRTUALVIEWSIZE
 */
const UINT32 vislib::net::cluster::AbstractControllerNode::DIRTY_VIRTUALVIEWSIZE
    = 0x00001000;