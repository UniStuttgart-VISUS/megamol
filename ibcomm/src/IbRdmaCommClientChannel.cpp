/*
 * IbRdmaCommClientChannel.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/IbRdmaCommClientChannel.h"

#include <cerrno>

#include "vislib/IbRdmaException.h"
#include "vislib/IPCommEndPoint.h"
#include "vislib/memutils.h"
#include "vislib/StackTrace.h"




/*
 * vislib::net::ib::IbRdmaCommClientChannel::Close
 */
void vislib::net::ib::IbRdmaCommClientChannel::Close(void) {
    VLSTACKTRACE("IbRdmaCommClientChannel::Close", __FILE__, __LINE__);

    int result = 0;                     // RDMA API results.

    result = ::rdma_disconnect(this->id);
    if (result != 0) {
        throw IbRdmaException("rdma_disconnect", errno, __FILE__, __LINE__);
    }

    // TODO
    ::rdma_dereg_mr(this->mrRecv);
    ::rdma_dereg_mr(this->mrSend);
    ::rdma_destroy_ep(this->id);
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::Connect
 */
void vislib::net::ib::IbRdmaCommClientChannel::Connect(
        SmartRef<AbstractCommEndPoint> endPoint) {
    VLSTACKTRACE("IbRdmaCommClientChannel::Connect", __FILE__, __LINE__);

    int result = 0;                     // RDMA API results.
    StringA node;                       // The address as string.
    StringA service;                    // Endpoint port as string.
    struct rdma_addrinfo hints;         // Input for getaddrinfo.
    struct rdma_addrinfo *addrInfo;     // Output of getaddrinfo.
    struct ibv_qp_init_attr attr;       // Queue pair properties.

    /* Format the address as node name and  the port number as service name. */
    IPCommEndPoint *cep = endPoint.DynamicPeek<IPCommEndPoint>();
    IPEndPoint& ep = static_cast<IPEndPoint&>(*cep);
    node = ep.GetIPAddress().ToStringA();
    service.Format("%d", ep.GetPort());

    /* Initialise our request. */
    ::ZeroMemory(&hints, sizeof(hints));
    hints.ai_port_space = RDMA_PS_TCP;

    /* Get the result. */
    result = ::rdma_getaddrinfo(const_cast<char *>(node.PeekBuffer()), 
        const_cast<char *>(service.PeekBuffer()), &hints, &addrInfo);
    if (result != 0) {
        throw IbRdmaException("rdma_getaddrinfo", errno, __FILE__, __LINE__);
    }

    /* Create the end point. */
    ::ZeroMemory(&attr, sizeof(attr));
    attr.cap.max_send_wr = 1;
    attr.cap.max_recv_wr = 1;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.cap.max_inline_data = 16;  // TODO
    attr.sq_sig_all = 1;

    attr.qp_context = this->id;

    result = ::rdma_create_ep(&this->id, addrInfo, NULL, &attr);
    ::rdma_freeaddrinfo(addrInfo);
    if (result != 0) {
        throw IbRdmaException("rdma_create_ep", errno, __FILE__, __LINE__);
    }

    this->mrRecv = ::rdma_reg_msgs(this->id, this->bufRecv, 
        this->cntBufRecv);
    if (this->mrRecv == NULL) {
        throw IbRdmaException("rdma_reg_msgs", errno, __FILE__, __LINE__);
    }

    this->mrSend = ::rdma_reg_msgs(this->id, this->bufSend, 
        this->cntBufSend);
    if (this->mrSend == NULL) {
        throw IbRdmaException("rdma_reg_msgs", errno, __FILE__, __LINE__);
    }

    result = ::rdma_post_recv(this->id, NULL, this->bufRecv, 
        this->cntBufRecv, this->mrRecv);
    if (result != 0) {
        throw Exception("TODO rdma_post_recv", __FILE__, __LINE__);
    }

    result = ::rdma_connect(this->id, NULL);
    if (result != 0) {
        throw Exception("TODO rdma_connect", __FILE__, __LINE__);
    }

}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::GetLocalEndPoint
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint> 
vislib::net::ib::IbRdmaCommClientChannel::GetLocalEndPoint(void) const {
    VLSTACKTRACE("IbRdmaCommClientChannel::GetLocalEndPoint", __FILE__, __LINE__);
    throw 1;    // TODO
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::GetRemoteEndPoint
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint>
vislib::net::ib::IbRdmaCommClientChannel::GetRemoteEndPoint(void) const {
    VLSTACKTRACE("IbRdmaCommClientChannel::GetRemoteEndPoint", __FILE__, __LINE__);
    throw 1;    // TODO
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::Receive
 */
SIZE_T vislib::net::ib::IbRdmaCommClientChannel::Receive(void *outData, 
        const SIZE_T cntBytes, const UINT timeout, const bool forceReceive) {
    VLSTACKTRACE("IbRdmaCommClientChannel::Receive", __FILE__, __LINE__);
    throw 1;    // TODO
}



/*
 * vislib::net::ib::IbRdmaCommClientChannel::Send
 */
SIZE_T vislib::net::ib::IbRdmaCommClientChannel::Send(const void *data, 
        const SIZE_T cntBytes, const UINT timeout,  const bool forceSend) {
    VLSTACKTRACE("IbRdmaCommClientChannel::Send", __FILE__, __LINE__);
    throw 1;    // TODO
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::IbRdmaCommClientChannel
 */
vislib::net::ib::IbRdmaCommClientChannel::IbRdmaCommClientChannel(void) 
        : bufRecv(NULL), bufSend(NULL), cntBufRecv(0), cntBufSend(0), id(NULL), 
        mrRecv(NULL), mrSend(NULL) {
    VLSTACKTRACE("IbRdmaCommClientChannel::IbRdmaCommClientChannel", __FILE__, __LINE__);

    this->cntBufRecv = this->cntBufSend = 1024; // TODO

    this->bufRecv = new BYTE[this->cntBufRecv];
    this->bufSend = new BYTE[this->cntBufSend];
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::~IbRdmaCommClientChannel
 */
vislib::net::ib::IbRdmaCommClientChannel::~IbRdmaCommClientChannel(void) {
    VLSTACKTRACE("IbRdmaCommClientChannel::~IbRdmaCommClientChannel", __FILE__, __LINE__);
}
