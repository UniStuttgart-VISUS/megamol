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
#include "vislib/StackTrace.h"


/*
 * vislib::net::ib::IbRdmaCommServerChannel::Create
 */
vislib::SmartRef<vislib::net::ib::IbRdmaCommServerChannel> 
vislib::net::ib::IbRdmaCommServerChannel::Create(void) {
    VLSTACKTRACE("IbRdmaCommServerChannel::Create", __FILE__, __LINE__);
    return SmartRef<IbRdmaCommServerChannel>(new IbRdmaCommServerChannel(), 
        false);
}


/*
 * vislib::net::ib::IbRdmaCommServerChannel::Accept
 */
vislib::SmartRef<vislib::net::AbstractCommClientChannel> 
vislib::net::ib::IbRdmaCommServerChannel::Accept(void) {
    VLSTACKTRACE("IbRdmaCommClientChannel::Accept", __FILE__, __LINE__);

    int result = 0;                     // RDMA API results.
    SmartRef<IbRdmaCommClientChannel> retval(new IbRdmaCommClientChannel(), 
        false);                         // Client channel.

    result = ::rdma_get_request(this->id, &retval->id);
    if (result != 0) {
        throw IbRdmaException("rdma_get_request", errno, __FILE__, __LINE__);
    }

    retval->createBuffers(1024, 1024);  // TODO

    retval->mrRecv = ::rdma_reg_msgs(retval->id, retval->bufRecv, 
        retval->cntBufRecv);
    if (retval->mrRecv == NULL) {
        throw IbRdmaException("rdma_reg_msgs", errno, __FILE__, __LINE__);
    }

    retval->mrSend = ::rdma_reg_msgs(retval->id, retval->bufSend, 
        retval->cntBufSend);
    if (retval->mrSend == NULL) {
        throw IbRdmaException("rdma_reg_msgs", errno, __FILE__, __LINE__);
    }

    //result = ::rdma_post_recv(retval->id, NULL, retval->bufRecv, 
    //    retval->cntBufRecv, retval->mrRecv);
    //if (result != 0) {
    //    throw Exception("TODO rdma_post_recv", __FILE__, __LINE__);
    //}

    result = ::rdma_accept(retval->id, NULL);
    if (result != 0) {
        throw Exception("TODO rdma_accept", __FILE__, __LINE__);
    }

    //result = ::rdma_get_recv_comp(retval->id, &retval->wc);
    //if (result != 0) {
    //    throw Exception("TODO rdma_get_recv_comp", __FILE__, __LINE__);
    //}

    return retval.DynamicCast<AbstractCommClientChannel>();
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
    hints.ai_port_space = RDMA_PS_TCP;
    hints.ai_flags = RAI_PASSIVE;

    /* Get the result. */
    result = ::rdma_getaddrinfo(NULL, const_cast<char *>(service.PeekBuffer()),
        &hints, &addrInfo);
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

    //attr.qp_context = this->id;

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
        throw IbRdmaException("rdma_disconnect", errno, __FILE__, __LINE__);
    }

    // TODO
//    ::rdma_dereg_mr(this->mr);
    ::rdma_destroy_ep(this->id);
}


/*
 * vislib::net::ib::IbRdmaCommServerChannel::GetLocalEndPoint
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint>
vislib::net::ib::IbRdmaCommServerChannel::GetLocalEndPoint(void) const {
    VLSTACKTRACE("IbRdmaCommServerChannel::GetLocalEndPoint", __FILE__, __LINE__);
    throw 1;    // TODO
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
vislib::net::ib::IbRdmaCommServerChannel::IbRdmaCommServerChannel(void) 
        : Super(), id(NULL) {
    VLSTACKTRACE("IbRdmaCommServerChannel::IbRdmaCommServerChannel", 
        __FILE__, __LINE__);
}


/*
 * vislib::net::ib::IbRdmaCommServerChannel::~IbRdmaCommServerChannel
 */
vislib::net::ib::IbRdmaCommServerChannel::~IbRdmaCommServerChannel(void) {
    VLSTACKTRACE("IbRdmaCommServerChannel::~IbRdmaCommServerChannel", 
        __FILE__, __LINE__);
}
