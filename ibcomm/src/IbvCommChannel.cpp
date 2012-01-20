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
#include "vislib/sysfunctions.h"
#include "vislib/Trace.h"


/*
 * vislib::net::ib::IbvCommChannel::Accept
 */
vislib::SmartRef<vislib::net::AbstractCommChannel> 
vislib::net::ib::IbvCommChannel::Accept(void) {
    VLSTACKTRACE("IbvCommChannel::Bind", __FILE__, __LINE__);





    WV_CONNECT_PARAM conParam;
    ::ZeroMemory(&conParam, sizeof(conParam));

    OVERLAPPED overlapped;
    ::ZeroMemory(&overlapped, sizeof(overlapped));
    overlapped.hEvent = ::CreateEvent(NULL, TRUE, FALSE, NULL);

    this->connectEndPoint->Accept(this->queuePair, &conParam, &overlapped);

    ::WaitForSingleObject(overlapped.hEvent, INFINITE);

    return vislib::SmartRef<vislib::net::AbstractCommChannel>(NULL);
}


/*
 * vislib::net::ib::IbvCommChannel::Bind
 */
void vislib::net::ib::IbvCommChannel::Bind(
        SmartRef<AbstractCommEndPoint> endPoint) {
    VLSTACKTRACE("IbvCommChannel::Bind", __FILE__, __LINE__);
    ////int result = 0;                     // RDMA CMA return values.

    ////this->evtChannel = ::rdma_create_event_channel();
    ////if (this->evtChannel == NULL) {
    ////    throw Exception("TODO: Creation of RDMA event channel failed", __FILE__, __LINE__);
    ////}

    ////result = ::rdma_create_id(this->evtChannel, &this->id, this, RDMA_PS_TCP);
    ////if (result != 0) {
    ////    throw Exception("TODO: Allocation of RMDA communication identifiert failed", __FILE__, __LINE__);
    ////}

    //IPCommEndPoint *cep = endPoint.DynamicPeek<IPCommEndPoint>();
    //IPEndPoint& ep = static_cast<IPEndPoint&>(*cep);
    ////::rdma_bind_addr(this->id, static_cast<struct sockaddr *>(ep));
    ////if (result != 0) {
    ////    throw Exception("TODO: Binding server address failed", __FILE__, __LINE__);
    ////}

    //VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Creating IB end point...\n");
    //if (FAILED(this->wvProvider->CreateConnectEndpoint(&this->connectEndPoint))) {
    //    throw Exception("TODO: CreateConnectEndpoint failed", __FILE__, __LINE__);
    //}

    HRESULT hr = S_OK;
    this->initialise();

    IPCommEndPoint *cep = endPoint.DynamicPeek<IPCommEndPoint>();
    IPEndPoint& ep = static_cast<IPEndPoint&>(*cep);

    VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Binding IB end point to %s...\n",
        ep.ToStringA().PeekBuffer());
    if (FAILED(hr = this->connectEndPoint->BindAddress(
            static_cast<struct sockaddr *>(ep)))) {
        throw sys::COMException(hr, __FILE__, __LINE__);
    }

    this->createQueuePair();
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


    VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Setting IB end point into "
        "listening state...\n");
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
        : connectEndPoint(NULL),  device(NULL), protectionDomain(NULL), 
        queuePair(NULL), wvProvider(NULL) {
    VLSTACKTRACE("IbvCommChannel::IbvCommChannel", __FILE__, __LINE__);
}


/*
 * vislib::net::ib::IbvCommChannel::~IbvCommChannel
 */
vislib::net::ib::IbvCommChannel::~IbvCommChannel(void) {
    VLSTACKTRACE("IbvCommChannel::~IbvCommChannel", __FILE__, __LINE__);

    sys::SafeRelease(this->connectEndPoint);
    sys::SafeRelease(this->device);
    sys::SafeRelease(this->protectionDomain);
    sys::SafeRelease(this->queuePair);
    sys::SafeRelease(this->wvProvider);
}


/*
 * vislib::net::ib::IbvCommChannel::createQueuePair
 */
void vislib::net::ib::IbvCommChannel::createQueuePair(void) {
    VLSTACKTRACE("IbvCommChannel::createQueuePair", __FILE__, __LINE__);
    WV_CONNECT_ATTRIBUTES connAttribs;	// Receives attributes of end point.
    HRESULT hr = S_OK;                  // Result of API calls.
    WV_QP_CREATE qpCreate;              // Creation parameter for queue pair.

    if (this->queuePair == NULL) {

        /* Get attributes of end point to get the device used for it. */
        ::ZeroMemory(&connAttribs, sizeof(connAttribs));
        VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Querying attributes of "
            "connect end point...\n");
        ASSERT(this->connectEndPoint != NULL);
        if (FAILED(hr = this->connectEndPoint->Query(&connAttribs))) {
            VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Querying attributes of "
                "connect end point failed with error code %d.\n", hr);
            throw sys::COMException(hr, __FILE__, __LINE__);
        }

        VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Opening IB device...\n");
        ASSERT(this->wvProvider != NULL);
        ASSERT(this->device == NULL);
        if (FAILED(hr = this->wvProvider->OpenDevice(
                connAttribs.Device.DeviceGuid, &this->device))) {
            VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Opening IB device failed "
                "with error code %d.\n", hr);
            throw sys::COMException(hr, __FILE__, __LINE__);
        }

        /* Allocate a protection domain for the device. */
        VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Allocating protection "
            "domain...\n");
        ASSERT (this->protectionDomain == NULL);
        ASSERT(this->device != NULL);
        if (FAILED(hr = this->device->AllocateProtectionDomain(
                &this->protectionDomain))) {
            VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Allocating protection "
                "domain failed with error code %d.\n", hr);
            throw sys::COMException(hr, __FILE__, __LINE__);
        }

        /* Allocate the queue pair. */
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

        VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Creating queue pair...\n");
        ASSERT(this->queuePair == NULL);
        if (FAILED(hr = this->protectionDomain->CreateConnectQueuePair(
            &qpCreate, &this->queuePair))) {
            VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Creating queue pair failed "
                "with error code %d.\n", hr);
            throw sys::COMException(hr, __FILE__, __LINE__);
        }

    } /* end if (!this->isInitialised) */
}


/*
 * vislib::net::ib::IbvCommChannel::initialise
 */
void vislib::net::ib::IbvCommChannel::initialise(void) {
    VLSTACKTRACE("IbvCommChannel::initialise", __FILE__, __LINE__);
    HRESULT hr = S_OK;      // Result of API calls.

    if (this->wvProvider == NULL) {
        VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Acquiring WinVerbs "
            "provider...\n");
        if (FAILED(hr = ::WvGetObject(IID_IWVProvider, 
                reinterpret_cast<void **>(&this->wvProvider)))) {
            VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Acquiring WinVerbs "
                "provider failed with error code %d.\n", hr);
            throw sys::COMException(hr, __FILE__, __LINE__);    
        }

        /* Create a connect end point. */
        VLTRACE(vislib::Trace::LEVEL_VL_VERBOSE, "Creating IB connect end "
            "point...\n");
        ASSERT(this->connectEndPoint == NULL);
        if (FAILED(hr = this->wvProvider->CreateConnectEndpoint(
                &this->connectEndPoint))) {
            VLTRACE(vislib::Trace::LEVEL_VL_ERROR, "Creating IB connect end "
                "point failed with error code %d.\n", hr);
            throw sys::COMException(hr, __FILE__, __LINE__);
        }
    } 

    ASSERT(this->wvProvider != NULL);
    ASSERT(this->connectEndPoint != NULL);
}
