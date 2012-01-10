/*  
 * IbvCommChannel.cpp
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/IbvCommChannel.h"

#include "vislib/assert.h"
#include "vislib/Exception.h"
#include "vislib/IPCommEndPoint.h"
#include "vislib/memutils.h"
#include "vislib/MissingImplementationException.h"


/*
 * vislib::net::ib::IbvCommChannel::Accept
 */
vislib::SmartRef<vislib::net::AbstractCommChannel> 
vislib::net::ib::IbvCommChannel::Accept(void) {
    VLSTACKTRACE("IbvCommChannel::Bind", __FILE__, __LINE__);
    ASSERT(this->wvProvider != NULL);
    ASSERT(this->connectEndPoint != NULL);


    //this->wvProvider->

    WV_CONNECT_ATTRIBUTES connAttribs;
    ::ZeroMemory(&connAttribs, sizeof(connAttribs));
    this->connectEndPoint->Query(&connAttribs);

    IWVDevice *device = NULL;
    this->wvProvider->OpenDevice(connAttribs.Device.DeviceGuid, &device);

    IWVProtectionDomain *protectionDomain = NULL;
    device->AllocateProtectionDomain(&protectionDomain);

    WV_QP_CREATE qpCreate;
    ::ZeroMemory(&qpCreate, sizeof(qpCreate));

    qpCreate.QpType = WvQpTypeRc;

    //qp_attr->send_cq = s_ctx->cq;
    //qp_attr->recv_cq = s_ctx->cq;
    //qp_attr->qp_type = IBV_QPT_RC;
    //qp_attr->cap.max_send_wr = 10;
    //qp_attr->cap.max_recv_wr = 10;
    //qp_attr->cap.max_send_sge = 1;
    //qp_attr->cap.max_recv_sge = 1;

    //create.pSendCq = qp_init_attr->send_cq->handle;
    //create.pReceiveCq = qp_init_attr->recv_cq->handle;
    //create.pSharedReceiveQueue = (qp_init_attr->srq != NULL) ?
    //							 qp_init_attr->srq->handle : NULL;
    //create.Context = qp;
    //create.SendDepth = qp_init_attr->cap.max_send_wr;
    //create.SendSge = qp_init_attr->cap.max_send_sge;
    //create.ReceiveDepth = qp_init_attr->cap.max_recv_wr;
    //create.ReceiveSge = qp_init_attr->cap.max_recv_sge;
    //create.MaxInlineSend = qp_init_attr->cap.max_inline_data;
    //create.InitiatorDepth = 0;
    //create.ResponderResources = 0;
    //create.QpType = (WV_QP_TYPE) qp_init_attr->qp_type;
    //create.QpFlags = qp_init_attr->sq_sig_all ? WV_QP_SIGNAL_SENDS : 0;

    IWVConnectQueuePair *qp;
    protectionDomain->CreateConnectQueuePair(&qpCreate, &qp);

    WV_CONNECT_PARAM conParam;
    ::ZeroMemory(&conParam, sizeof(conParam));

    OVERLAPPED overlapped;
    ::ZeroMemory(&overlapped, sizeof(overlapped));
    overlapped.hEvent = ::CreateEvent(NULL, TRUE, FALSE, NULL);
    
    this->connectEndPoint->Accept(qp, &conParam, &overlapped);

    ::WaitForSingleObject(overlapped.hEvent, INFINITE);

    return vislib::SmartRef<vislib::net::AbstractCommChannel>(NULL);
}


/*
 * vislib::net::ib::IbvCommChannel::Bind
 */
void vislib::net::ib::IbvCommChannel::Bind(
        SmartRef<AbstractCommEndPoint> endPoint) {
    VLSTACKTRACE("IbvCommChannel::Bind", __FILE__, __LINE__);
    //int result = 0;                     // RDMA CMA return values.

    //this->evtChannel = ::rdma_create_event_channel();
    //if (this->evtChannel == NULL) {
    //    throw Exception("TODO: Creation of RDMA event channel failed", __FILE__, __LINE__);
    //}

    //result = ::rdma_create_id(this->evtChannel, &this->id, this, RDMA_PS_TCP);
    //if (result != 0) {
    //    throw Exception("TODO: Allocation of RMDA communication identifiert failed", __FILE__, __LINE__);
    //}

    IPCommEndPoint *cep = endPoint.DynamicPeek<IPCommEndPoint>();
    IPEndPoint& ep = static_cast<IPEndPoint&>(*cep);
    //::rdma_bind_addr(this->id, static_cast<struct sockaddr *>(ep));
    //if (result != 0) {
    //    throw Exception("TODO: Binding server address failed", __FILE__, __LINE__);
    //}

    if (FAILED(this->wvProvider->CreateConnectEndpoint(&this->connectEndPoint))) {
        throw Exception("TODO: CreateConnectEndpoint failed", __FILE__, __LINE__);
    }

    if (FAILED(this->connectEndPoint->BindAddress(static_cast<struct sockaddr *>(ep)))) {
        throw Exception("TODO: BindAddress failed", __FILE__, __LINE__);
    }

}


/*
 * vislib::net::ib::IbvCommChannel::Close
 */
void vislib::net::ib::IbvCommChannel::Close(void) {
    VLSTACKTRACE("IbvCommChannel::Close", __FILE__, __LINE__);
    // TODO
}


/*
 * vislib::net::ib::IbvCommChannel::Connect
 */
void vislib::net::ib::IbvCommChannel::Connect(
        SmartRef<AbstractCommEndPoint> endPoint) {
    VLSTACKTRACE("IbvCommChannel::Connect", __FILE__, __LINE__);
    // TODO
}


/*
 * vislib::net::ib::IbvCommChannel::GetLocalEndPoint
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::ib::IbvCommChannel::GetLocalEndPoint(void) const {
    VLSTACKTRACE("IbvCommChannel::GetLocalEndPoint", __FILE__, __LINE__);
    // TODO
    throw vislib::MissingImplementationException("GetLocalEndPoint", __FILE__, __LINE__);
}


/*
 * vislib::net::ib::IbvCommChannel::GetRemoteEndPoint
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::ib::IbvCommChannel::GetRemoteEndPoint(void) const {
    VLSTACKTRACE("IbvCommChannel::GetRemoteEndPoint", __FILE__, __LINE__);
    // TODO
    throw vislib::MissingImplementationException("GetLocalEndPoint", __FILE__, __LINE__);
}


/*
 * vislib::net::ib::IbvCommChannel::Listen
 */
void vislib::net::ib::IbvCommChannel::Listen(const int backlog) {
    VLSTACKTRACE("IbvCommChannel::Listen", __FILE__, __LINE__);
    //int result = 0;                     // RDMA CMA return values.
    //struct rdma_cm_event *evt;          // RDMA event received.
    //struct rdma_cm_event evtCopy;       // Copy of event for processing.

    //while (::rdma_get_cm_event(this->evtChannel, &evt) == 0) {
    //    ::memcpy(&evtCopy, evt, sizeof(*evt));
    //    ::rdma_ack_cm_event(evt);
    //    
    //    switch (evt->event) {
    //        case RDMA_CM_EVENT_CONNECT_REQUEST:
    //            this->onConnectRequest(evt->id);
    //            break;

    //        case RDMA_CM_EVENT_ESTABLISHED:
    //            break;

    //        case RDMA_CM_EVENT_DISCONNECTED:
    //            break;

    //        default:
    //            throw Exception("TODO: Unexpected event.", __FILE__, __LINE__);
    //    }
    //}

    if (FAILED(this->connectEndPoint->Listen(backlog))) {
        throw Exception("TODO: Listen failed", __FILE__, __LINE__);
    }
}


/* 
 * vislib::net::ib::IbvCommChannel::Receive
 */
SIZE_T vislib::net::ib::IbvCommChannel::Receive(void *outData, 
        const SIZE_T cntBytes, const UINT timeout,  const bool forceReceive) {
    // TODO
    return 0;
}


/* 
 * vislib::net::ib::IbvCommChannel::Receive
 */
SIZE_T vislib::net::ib::IbvCommChannel::Send(const void *data, 
        const SIZE_T cntBytes, const UINT timeout, const bool forceSend) {
    // TODO
    return 0;
}


/*
 * vislib::net::ib::IbvCommChannel::IbvCommChannel
 */
vislib::net::ib::IbvCommChannel::IbvCommChannel(void) 
        : connectEndPoint(NULL),  wvProvider(NULL) {
    VLSTACKTRACE("IbvCommChannel::IbvCommChannel", __FILE__, __LINE__);

    if (FAILED(::WvGetObject(IID_IWVProvider, reinterpret_cast<void **>(&this->wvProvider)))) {
        throw Exception("TODO: Could not get WV provider!", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::ib::IbvCommChannel::~IbvCommChannel
 */
vislib::net::ib::IbvCommChannel::~IbvCommChannel(void) {
    VLSTACKTRACE("IbvCommChannel::~IbvCommChannel", __FILE__, __LINE__);
}

