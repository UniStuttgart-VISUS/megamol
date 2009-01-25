/*
 * AbstractControllerNode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractControllerNode.h"

#include "vislib/RawStorage.h"
#include "vislib/RawStorageSerialiser.h"
#include "vislib/Trace.h"
#include "vislib/unreferenced.h"


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
    return false;
}


/*
 * vislib::net::cluster::AbstractControllerNode::onPeerConnected
 */
void vislib::net::cluster::AbstractControllerNode::onPeerConnected(
        const PeerIdentifier& peerId) throw() {
    try {
        this->sendAllParameters(&peerId);
    } catch (Exception& e) {
        VL_DBGONLY_REFERENCED_LOCAL_VARIABLE(e);
        VLTRACE(Trace::LEVEL_VL_ERROR, "Sending camera parameters to newly "
            "connected node failed: %s\n", e.GetMsgA());
    } catch (...) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Sending camera parameters to newly "
            "connected node failed for a unknown reason\n");
    }
}


/*
 * vislib::net::cluster::AbstractControllerNode::sendAllParameters
 */
void vislib::net::cluster::AbstractControllerNode::sendAllParameters(
        const PeerIdentifier *peerId) {
    RawStorage msg;
    RawStorageSerialiser serialiser(&msg);

    /* 
     * Build a parameter message into 'msg' that looks like this:
     *
     * |-----------------------------------------|
     * | MsgHeader                               |
     * |-----------------------------------------|
     * | BlockHeader for camera parameter limits |
     * |-----------------------------------------|
     * | Serialised camera parameter limits      |
     * |-----------------------------------------|
     * | BlockHeader for camera parameters       |
     * |-----------------------------------------|
     * | Serialised camera parameters            |
     * |-----------------------------------------|
     */

    /* Serialise the camera parameter limits first. */
    msg.EnforceSize(0);
    ASSERT(msg.GetSize() == 0);     // Storage must be empty!
    serialiser.SetOffset(sizeof(MessageHeader) + sizeof(BlockHeader));
    this->parameters->Limits()->Serialise(serialiser);

    BlockHeader *blkHdr1 = msg.AsAt<BlockHeader>(sizeof(MessageHeader));
    blkHdr1->BlockId = MSGID_CAM_LIMITS;
    blkHdr1->BlockLength = static_cast<UINT32>(msg.GetSize()
        - sizeof(MessageHeader)
        - sizeof(BlockHeader));
    VLTRACE(Trace::LEVEL_VL_VERBOSE, "Packaged %u B serialised camera parameter "
        "limits (Message %u).\n", blkHdr1->BlockLength, blkHdr1->BlockId);

    /* 
     * Serialise the parameters as a second block (deserialisation of parameters
     * will trigger evaluation against limits on client side).
     */
    ASSERT(msg.GetSize() >= serialiser.Offset());
    serialiser.SetOffset(static_cast<UINT32>(msg.GetSize()
        + sizeof(BlockHeader)));
    this->parameters->Serialise(serialiser);

    blkHdr1 = msg.AsAt<BlockHeader>(sizeof(MessageHeader));
    BlockHeader *blkHdr2 = msg.AsAt<BlockHeader>(sizeof(MessageHeader)
        + sizeof(BlockHeader) + blkHdr1->BlockLength);
    blkHdr2->BlockId = MSGID_CAM_SERIALISEDCAMPARAMS;
    blkHdr2->BlockLength = static_cast<UINT32>(msg.GetSize()
        - sizeof(MessageHeader)
        - 2 * sizeof(BlockHeader)
        - blkHdr1->BlockLength);
    VLTRACE(Trace::LEVEL_VL_VERBOSE, "Packaged %u B serialised camera parameters "
        "(Message %u).\n", blkHdr2->BlockLength, blkHdr2->BlockId);

    /* Fill in the message header for a compound message. */
    ASSERT(blkHdr1->BlockId == MSGID_CAM_LIMITS);
    ASSERT(blkHdr2->BlockId == MSGID_CAM_SERIALISEDCAMPARAMS);

    MessageHeader *msgHdr = msg.As<MessageHeader>();
    InitialiseMessageHeader(*msgHdr);
    msgHdr->Header.BlockId = MSGID_MULTIPLE;
    msgHdr->Header.BlockLength = static_cast<UINT32>(msg.GetSize() 
        - sizeof(MessageHeader));
    ASSERT(msgHdr->Header.BlockLength 
        == sizeof(BlockHeader) + blkHdr1->BlockLength
        + sizeof(BlockHeader) + blkHdr2->BlockLength);

    /* Post the message. */
    if (peerId != NULL) {
        this->sendToPeer(*peerId, msg.As<BYTE>(), msg.GetSize());
    } else {
        this->sendToEachPeer(msg.As<BYTE>(), msg.GetSize());
    }
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
