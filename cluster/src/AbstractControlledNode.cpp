/*
 * AbstractControlledNode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractControlledNode.h"

#include "vislib/CameraParamsStore.h"
#include "vislib/clustermessages.h"
#include "vislib/RawStorage.h"
#include "vislib/RawStorageSerialiser.h"
#include "vislib/ShallowPoint.h"
#include "vislib/ShallowVector.h"
#include "vislib/Trace.h"



/*
 * vislib::net::cluster::AbstractControlledNode::~AbstractControlledNode
 */
vislib::net::cluster::AbstractControlledNode::~AbstractControlledNode(void) {
    // Nothing to do.
}


/*
 * vislib::net::cluster::AbstractControlledNode::AbstractControlledNode
 */
vislib::net::cluster::AbstractControlledNode::AbstractControlledNode(void) 
        : parameters(NULL) {
    this->parameters = new graphics::CameraParamsStore();
}


/*
 * vislib::net::cluster::AbstractControlledNode::onMessageReceived
 */
bool vislib::net::cluster::AbstractControlledNode::onMessageReceived(
        const Socket& src, const UINT msgId, const BYTE *body, 
        const SIZE_T cntBody) {
#define SET_AS_INTEGRAL_PARAM(type, name)                                      \
    this->parameters->Set##name(*reinterpret_cast<const type *>(body))
#define SET_AS_SHALLOW_PARAM(type, space, name) {                              \
    graphics::Shallow##space##type sh(                                         \
        reinterpret_cast<graphics::space##Type *>(const_cast<BYTE *>(body)));  \
    this->parameters->Set##name(sh);                                           \
    }

    switch (msgId) {

        case MSGID_CAM_APERTUREANGLE:
            SET_AS_INTEGRAL_PARAM(math::AngleDeg, ApertureAngle);
            return true;

        case MSGID_CAM_EYE:
            SET_AS_INTEGRAL_PARAM(graphics::CameraParameters::StereoEye, Eye);
            return true;

        case MSGID_CAM_FARCLIP:
            SET_AS_INTEGRAL_PARAM(graphics::SceneSpaceType, FarClip);
            return true;

        case MSGID_CAM_FOCALDISTANCE:
            SET_AS_INTEGRAL_PARAM(graphics::SceneSpaceType, FocalDistance);
            return true;

        case MSGID_CAM_LIMITS: {
            RawStorageSerialiser serialiser(body, cntBody);
            this->parameters->Limits()->Deserialise(serialiser);
            } 
            return true;

        case MSGID_CAM_LOOKAT:
            SET_AS_SHALLOW_PARAM(Point3D, SceneSpace, LookAt);
            return true;

        case MSGID_CAM_NEARCLIP:
            SET_AS_INTEGRAL_PARAM(graphics::SceneSpaceType, NearClip);
            return true;

        case MSGID_CAM_POSITION:
            SET_AS_SHALLOW_PARAM(Point3D, SceneSpace, Position);
            return true;

        case MSGID_CAM_PROJECTION:
            SET_AS_INTEGRAL_PARAM(graphics::CameraParameters::ProjectionType,
                Projection);
            return true;

        case MSGID_CAM_STEREODISPARITY:
            SET_AS_INTEGRAL_PARAM(graphics::SceneSpaceType, StereoDisparity);
            return true;

        case MSGID_CAM_TILERECT:
            SET_AS_SHALLOW_PARAM(Rectangle, ImageSpace, TileRect);
            return true;

        case MSGID_CAM_UP:
            SET_AS_SHALLOW_PARAM(Vector3D, SceneSpace, Up);
            return true;

        case MSGID_CAM_VIRTUALVIEWSIZE:
            SET_AS_SHALLOW_PARAM(Dimension2D, ImageSpace, VirtualViewSize);
            return true;

        case MSGID_CAM_SERIALISEDCAMPARAMS: {
            RawStorageSerialiser serialiser(body, cntBody);
            this->parameters->Deserialise(serialiser);
            } 
            return true;

        default:
            return false;
    }

#undef SET_AS_INTEGRAL_PARAM
#undef SET_AS_SHALLOW_PARAM
}


/*
 * vislib::net::cluster::AbstractControlledNode::operator =
 */
vislib::net::cluster::AbstractControlledNode& 
vislib::net::cluster::AbstractControlledNode::operator =(
        const AbstractControlledNode& rhs) {
    if (this != &rhs) {
        AbstractClusterNode::operator =(rhs);
        this->parameters = rhs.parameters;
    }
    return *this;
}
