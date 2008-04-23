/*
 * AbstractControllerNode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractControllerNode.h"

#include "vislib/RawStorage.h"
#include "vislib/RawStorageSerialiser.h"


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
    this->sendIntegralCamParam(MSGID_CAM_APERTUREANGLE, newValue);
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnEyeChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnEyeChanged(
       const graphics::CameraParameters::StereoEye newValue) {
    this->sendIntegralCamParam(MSGID_CAM_EYE, newValue);
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnFarClipChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnFarClipChanged(
        const graphics::SceneSpaceType newValue) {
    this->sendIntegralCamParam(MSGID_CAM_FARCLIP, newValue);
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnFocalDistanceChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnFocalDistanceChanged(
        const graphics::SceneSpaceType newValue) {
    this->sendIntegralCamParam(MSGID_CAM_FOCALDISTANCE, newValue);
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnLookAtChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnLookAtChanged(
        const graphics::SceneSpacePoint3D& newValue) {
    this->sendVectorialCamParam<graphics::SceneSpaceType, 3>(MSGID_CAM_LOOKAT,
        newValue.PeekCoordinates());
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnNearClipChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnNearClipChanged(
        const graphics::SceneSpaceType newValue) {
    this->sendIntegralCamParam(MSGID_CAM_NEARCLIP, newValue);
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnPositionChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnPositionChanged(
        const graphics::SceneSpacePoint3D& newValue) {
    this->sendVectorialCamParam<graphics::SceneSpaceType, 3>(MSGID_CAM_POSITION,
        newValue.PeekCoordinates());
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnProjectionChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnProjectionChanged(
        const graphics::CameraParameters::ProjectionType newValue) {
    this->sendIntegralCamParam(MSGID_CAM_PROJECTION, newValue);
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnStereoDisparityChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnStereoDisparityChanged(
        const graphics::SceneSpaceType newValue) {
    this->sendIntegralCamParam(MSGID_CAM_STEREODISPARITY, newValue);
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnTileRectChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnTileRectChanged(
        const graphics::ImageSpaceRectangle& newValue) {
    this->sendVectorialCamParam<graphics::ImageSpaceType, 4>(MSGID_CAM_TILERECT,
        newValue.PeekBounds());
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnUpChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnUpChanged(
        const graphics::SceneSpaceVector3D& newValue) {
    this->sendVectorialCamParam<graphics::SceneSpaceType, 3>(MSGID_CAM_UP,
        newValue.PeekComponents());
}


/* 
 * vislib::net::cluster::AbstractControllerNode::OnVirtualViewSizeChanged
 */
void vislib::net::cluster::AbstractControllerNode::OnVirtualViewSizeChanged(
        const graphics::ImageSpaceDimension& newValue) {
    this->sendVectorialCamParam<graphics::ImageSpaceType, 2>(
        MSGID_CAM_VIRTUALVIEWSIZE, newValue.PeekDimension());
}


/*
 * vislib::net::cluster::AbstractControllerNode::AbstractControllerNode
 */
vislib::net::cluster::AbstractControllerNode::AbstractControllerNode(
        SmartPtr<graphics::CameraParameters> params) 
        : AbstractClusterNode(), graphics::CameraParameterObserver() {
    this->setParameters(params);
}


/*
 * vislib::net::cluster::AbstractControllerNode::AbstractControllerNode
 */
vislib::net::cluster::AbstractControllerNode::AbstractControllerNode(
        const AbstractControllerNode& rhs) 
        : AbstractClusterNode(rhs), graphics::CameraParameterObserver(rhs) {
    this->setParameters(rhs.parameters);
}


/*
 * vislib::net::cluster::AbstractControllerNode::onMessageReceived
 */
bool vislib::net::cluster::AbstractControllerNode::onMessageReceived(
        const Socket& src, const UINT msgId, const BYTE *body, 
        const SIZE_T cntBody) {
    switch (msgId) {
        case MSGID_INTRODUCE:
            // TODO: Send all parameters to specified client.
            return false;

        default:
            return false;
    }
}


/*
 * vislib::net::cluster::AbstractControllerNode::sendAllParameters
 */
void vislib::net::cluster::AbstractControllerNode::sendAllParameters(void) {
    RawStorage msg;
    RawStorageSerialiser serialiser(&msg);

    /* 
     * Build a parameter message into 'msg' that looks like this:
     *
     * |-----------------------------------------|
     * | MsgHeader                               |
     * |-----------------------------------------|
     * | BlockHeader for camera parameters       |
     * |-----------------------------------------|
     * | Serialised camera parameters            |
     * |-----------------------------------------|
     * | BlockHeader for camera parameter limits |
     * |-----------------------------------------|
     * | Serialiser camera parameter limits      |
     * |-----------------------------------------|
     */

    /* Serialise the camera parameters. */
    msg.EnforceSize(0);
    ASSERT(msg.GetSize() == 0);
    serialiser.SetOffset(sizeof(MessageHeader) + sizeof(BlockHeader));
    this->parameters->Serialise(serialiser);
    
    BlockHeader *blkHdr1 = msg.AsAt<BlockHeader>(sizeof(MessageHeader));
    blkHdr1->BlockId = MSGID_CAM_SERIALISEDCAMPARAMS;
    blkHdr1->BlockLength = static_cast<UINT32>(msg.GetSize() 
        - sizeof(MessageHeader) - sizeof(BlockHeader));
    
    /* Serialise the parameter limits as a second block. */
    ASSERT(msg.GetSize() > serialiser.Offset());
    serialiser.SetOffset(static_cast<UINT32>(msg.GetSize() 
        + sizeof(BlockHeader)));
    this->parameters->Limits()->Serialise(serialiser);

    BlockHeader *blkHdr2 = msg.AsAt<BlockHeader>(sizeof(MessageHeader)
        + sizeof(BlockHeader) + blkHdr1->BlockLength);
    blkHdr2->BlockId = MSGID_CAM_LIMITS;
    blkHdr2->BlockLength = static_cast<UINT32>(msg.GetSize() 
        - sizeof(MessageHeader) - 2 * sizeof(BlockHeader) 
        - blkHdr1->BlockLength);

    /* Fill in the message header for a compound message. */
    MessageHeader *msgHdr = msg.As<MessageHeader>();
    InitialiseMessageHeader(*msgHdr);
    msgHdr->Header.BlockId = MSGID_MULTIPLE;
    msgHdr->Header.BlockLength = static_cast<UINT32>(msg.GetSize() 
        - sizeof(MessageHeader));
    ASSERT(msgHdr->Header.BlockLength == sizeof(MessageHeader)
        + sizeof(BlockHeader) + blkHdr1->BlockLength
        + sizeof(BlockHeader) + blkHdr2->BlockLength);

    /* Post the message. */
    this->sendToEachPeer(msg.As<BYTE>(), msg.GetSize());
}


/*
 * vislib::net::cluster::AbstractControllerNode::setParameters
 */
void vislib::net::cluster::AbstractControllerNode::setParameters(
        const SmartPtr<graphics::CameraParameters>& params) {
    if (!this->parameters.IsNull()) {
        this->getObservableParameters()->RemoveCameraParameterObserver(this);
    }
    if (params.DynamicCast<graphics::ObservableCameraParams>() == NULL) {
        throw IllegalParamException("params", __FILE__, __LINE__);
    }

    this->parameters = params;
    this->getObservableParameters()->AddCameraParameterObserver(this);
}


/*
 * vislib::net::cluster::AbstractControllerNode::operator =
 */
vislib::net::cluster::AbstractControllerNode& 
vislib::net::cluster::AbstractControllerNode::operator =(
        const AbstractControllerNode& rhs) {
    if (this != &rhs) {
        AbstractClusterNode::operator =(rhs);
        graphics::CameraParameterObserver::operator =(rhs);
        this->setParameters(rhs.parameters);
    }
    return *this;
}
