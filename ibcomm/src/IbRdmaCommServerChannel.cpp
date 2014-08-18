/*
 * IbRdmaCommServerChannel.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/IbRdmaCommServerChannel.h"

#include <cerrno>

#include "vislib/IbRdmaCommClientChannel.h"
#include "vislib/IPCommEndPoint.h"
#include "vislib/memutils.h"
#include "vislib/Trace.h"


/*
 * vislib::net::ib::IbRdmaCommServerChannel::Create
 */
vislib::SmartRef<vislib::net::ib::IbRdmaCommServerChannel> 
vislib::net::ib::IbRdmaCommServerChannel::Create(const SIZE_T cntBufRecv,
        const SIZE_T cntBufSend) {
    VLSTACKTRACE("IbRdmaCommServerChannel::Create", __FILE__, __LINE__);
    return SmartRef<IbRdmaCommServerChannel>(new IbRdmaCommServerChannel(
        cntBufRecv, cntBufSend), false);
}


/*
 * vislib::net::ib::IbRdmaCommServerChannel::Create
 */
vislib::SmartRef<vislib::net::ib::IbRdmaCommServerChannel> 
vislib::net::ib::IbRdmaCommServerChannel::Create(const SIZE_T cntBuf) {
    VLSTACKTRACE("IbRdmaCommServerChannel::Create", __FILE__, __LINE__);
    return SmartRef<IbRdmaCommServerChannel>(new IbRdmaCommServerChannel(
        cntBuf, cntBuf), false);
}


/*
 * vislib::net::ib::IbRdmaCommServerChannel::Accept
 */
vislib::SmartRef<vislib::net::AbstractCommClientChannel> 
vislib::net::ib::IbRdmaCommServerChannel::Accept(void) {
    VLSTACKTRACE("IbRdmaCommClientChannel::Accept", __FILE__, __LINE__);

    SmartRef<IbRdmaCommClientChannel> retval = this->Accept(
        NULL, this->cntBufRecv, NULL, this->cntBufSend);
    return retval.DynamicCast<AbstractCommClientChannel>();
}


/*
 * vislib::net::ib::IbRdmaCommServerChannel::Accept
 */
vislib::SmartRef<vislib::net::ib::IbRdmaCommClientChannel> 
vislib::net::ib::IbRdmaCommServerChannel::Accept(BYTE *bufRecv, 
        const SIZE_T cntBufRecv, BYTE *bufSend, const SIZE_T cntBufSend) {
    VLSTACKTRACE("IbRdmaCommClientChannel::Accept", __FILE__, __LINE__);

    int result = 0;                             // RDMA API results.
    SmartRef<IbRdmaCommClientChannel> retval;   // Client channel.

    // Allocate return value.
    retval = IbRdmaCommClientChannel::Create(bufRecv, cntBufRecv, 
        bufSend, cntBufSend);

    result = ::rdma_get_request(this->id, &retval->id);
    if (result != 0) {
        throw IbRdmaException("rdma_get_request", errno, __FILE__, __LINE__);
    }

    // HACK: There is a known bug in the RDMA library, which prevents 
    // rdma_get_request from creating the send and receive CQ for other than
    // the first client channels. See 
    // http://permalink.gmane.org/gmane.linux.drivers.rdma/6188 for more 
    // details. We circumvent this by re-creating the queue pair with our
    // saved initialisation attributes for any end point that does not have
    // a send or receive CQ.
    if ((retval->id->send_cq == NULL) || (retval->id->recv_cq == NULL)) {
        ASSERT(retval->id != NULL);
        ::rdma_destroy_qp(retval->id);

        struct ibv_qp_init_attr attr;
        ::memcpy(&attr, &this->qpAttr, sizeof(attr));
        result = ::rdma_create_qp(retval->id, NULL, &attr);
        if (result != 0) {
            throw IbRdmaException("rdma_create_qp", errno, __FILE__, __LINE__);
        }
    }

    retval->registerBuffers();

    // Post an initial receive before acceping the connection. This will ensure 
    // that the peer can directly start sending. We always keep an receive 
    // request in-flight.
    retval->postReceive();

    result = ::rdma_accept(retval->id, NULL);
    if (result != 0) {
        throw IbRdmaException("rdma_accept", errno, __FILE__, __LINE__);
    }

    //result = ::rdma_get_recv_comp(retval->id, &retval->wc);
    //if (result != 0) {
    //    throw Exception("TODO rdma_get_recv_comp", __FILE__, __LINE__);
    //}

    return retval;
}


/*
 * vislib::net::ib::IbRdmaCommServerChannel::Bind
 */
void vislib::net::ib::IbRdmaCommServerChannel::Bind(
        SmartRef<AbstractCommEndPoint> endPoint) {
    VLSTACKTRACE("IbRdmaCommClientChannel::Bind", __FILE__, __LINE__);

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
    hints.ai_port_space = RDMA_PS_TCP;  // RDMA_PS_UDP
    hints.ai_flags = RAI_PASSIVE;

    /* Get the result. */
    result = ::rdma_getaddrinfo(NULL, const_cast<char *>(service.PeekBuffer()),
        &hints, &addrInfo);
    if (result != 0) {
        throw IbRdmaException("rdma_getaddrinfo", errno, __FILE__, __LINE__);
    }

    // This copy is part of the hack in IbRdmaCommServerChannel::Accept().
    ::memcpy(&attr, &this->qpAttr, sizeof(attr));
    result = ::rdma_create_ep(&this->id, addrInfo, NULL, &attr);
    ::rdma_freeaddrinfo(addrInfo);
    if (result != 0) {
        throw IbRdmaException("rdma_create_ep", errno, __FILE__, __LINE__);
    }
}


/*
 * vislib::net::ib::IbRdmaCommServerChannel::Close
 */
void vislib::net::ib::IbRdmaCommServerChannel::Close(void) {
    VLSTACKTRACE("IbRdmaCommServerChannel::Close", __FILE__, __LINE__);

    int result = 0;                     // RDMA API results.

    result = ::rdma_disconnect(this->id);
    if (result != 0) {
        VLTRACE(Trace::LEVEL_VL_WARN, "rdma_disconnect failed with error "
            "code %d when closing the channel.\n", errno);
    }

    if (this->id != NULL) {
        ::rdma_destroy_ep(this->id);
        this->id = NULL;
    }
}


/*
 * vislib::net::ib::IbRdmaCommServerChannel::GetLocalEndPoint
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint>
vislib::net::ib::IbRdmaCommServerChannel::GetLocalEndPoint(void) const {
    VLSTACKTRACE("IbRdmaCommServerChannel::GetLocalEndPoint", __FILE__, __LINE__);
    WV_CONNECT_ATTRIBUTES attribs;
    this->id->ep.connect->Query(&attribs);
    return IPCommEndPoint::Create(attribs.LocalAddress.Sin);
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::Listen
 */
void vislib::net::ib::IbRdmaCommServerChannel::Listen(const int backlog) {
    VLSTACKTRACE("IbRdmaCommServerChannel::Listen", __FILE__, __LINE__);

    int result = 0;                     // RDMA API results.

    result = ::rdma_listen(this->id, 0);    // TODO: real backlog allocates Gigs of mem...
    if (result != 0) {
        throw IbRdmaException("rdma_listen", errno, __FILE__, __LINE__);
    }

}


/*
 * vislib::net::ib::IbRdmaCommServerChannel::IbRdmaCommServerChannel
 */
vislib::net::ib::IbRdmaCommServerChannel::IbRdmaCommServerChannel(
        const SIZE_T cntBufRecv, const SIZE_T cntBufSend) 
        : Super(), cntBufRecv(cntBufRecv), cntBufSend(cntBufSend), id(NULL) {
    VLSTACKTRACE("IbRdmaCommServerChannel::IbRdmaCommServerChannel", 
        __FILE__, __LINE__);

    // TODO: Move this?
    ::ZeroMemory(&this->qpAttr, sizeof(this->qpAttr));
    this->qpAttr.cap.max_send_wr = 1;
    this->qpAttr.cap.max_recv_wr = 1;
    this->qpAttr.cap.max_send_sge = 1;
    this->qpAttr.cap.max_recv_sge = 1;
    this->qpAttr.cap.max_inline_data = 0;  // TODO
    this->qpAttr.sq_sig_all = 1;
}


/*
 * vislib::net::ib::IbRdmaCommServerChannel::~IbRdmaCommServerChannel
 */
vislib::net::ib::IbRdmaCommServerChannel::~IbRdmaCommServerChannel(void) {
    VLSTACKTRACE("IbRdmaCommServerChannel::~IbRdmaCommServerChannel", 
        __FILE__, __LINE__);
    //try {
    //    this->Close();
    //} catch (...) {
    //    VLTRACE(Trace::LEVEL_VL_WARN, "An exception was caught when closing "
    //        "a IbRdmaCommServerChannel in its destructor.\n");
    //}
}
