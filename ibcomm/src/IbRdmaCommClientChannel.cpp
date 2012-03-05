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
#include "vislib/Trace.h"


/*
 * vislib::net::ib::IbRdmaCommClientChannel::Create
 */
vislib::SmartRef<vislib::net::ib::IbRdmaCommClientChannel> 
vislib::net::ib::IbRdmaCommClientChannel::Create(BYTE *bufRecv, 
        const SIZE_T cntBufRecv, BYTE *bufSend, const SIZE_T cntBufSend) {
    VLSTACKTRACE("IbRdmaCommClientChannel::Create", __FILE__, __LINE__);
    SmartRef<IbRdmaCommClientChannel> retval(new IbRdmaCommClientChannel(), 
        false);
    retval->setBuffers(bufRecv, cntBufRecv, bufSend, cntBufSend);
    return retval;
}


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
    // TODO: This is really evil, isn't it?

    if (this->mrRecv != NULL) {
        ::rdma_dereg_mr(this->mrRecv);
        this->mrRecv = NULL;
    }

    if (this->mrSend != NULL) {
        ::rdma_dereg_mr(this->mrSend);
        this->mrSend = NULL;
    }

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

    this->registerBuffers();

    // Post a first receive operation before connecting as we always need to 
    // have one in-flight receive. That is because the server is allowed to
    // send as first operation before receiving something.
    this->postReceive();

    result = ::rdma_connect(this->id, NULL);
    if (result != 0) {
        throw IbRdmaException("rdma_connect", errno, __FILE__, __LINE__);
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

    int result = 0;                     // RDMA API results.
    struct ibv_wc wc;                   // Receives the completion parameters.
    BYTE *out = static_cast<BYTE *>(outData);   // Cursor through output while copying.
    SIZE_T totalReceived = 0;           // # of bytes totally received.
    SIZE_T lastReceived = 0;            // # of bytes received in last op.

    do {
        while (!::ibv_poll_cq(this->id->recv_cq, 1, &wc));
        //result = ::rdma_get_recv_comp(this->id, &wc);
        //if (result != 0) {
        //    throw IbRdmaException("rdma_get_recv_comp", errno, __FILE__, __LINE__);
        //}

        lastReceived = (cntBytes < this->cntBufRecv) 
            ? cntBytes : this->cntBufRecv;

        ::memcpy(out, this->bufRecv, lastReceived);

        totalReceived += lastReceived;
        out += lastReceived;
            
        this->postReceive();

    } while (forceReceive && (totalReceived < cntBytes) && (lastReceived > 0));

    // TODO: Kann das das IB auch?!
    //throw PeerDisconnectedException(
    //    PeerDisconnectedException::FormatMessageForLocalEndpoint(
    //    this->socket.GetLocalEndPoint().ToStringW().PeekBuffer()), 
    //    __FILE__, __LINE__);

    return totalReceived;
}



/*
 * vislib::net::ib::IbRdmaCommClientChannel::Send
 */
SIZE_T vislib::net::ib::IbRdmaCommClientChannel::Send(const void *data, 
        const SIZE_T cntBytes, const UINT timeout,  const bool forceSend) {
    VLSTACKTRACE("IbRdmaCommClientChannel::Send", __FILE__, __LINE__);

    int result = 0;                     // RDMA API results.
    struct ibv_wc wc;                   // Receives the completion parameters.
    const BYTE *in = static_cast<const BYTE *>(data); // Cursor through input while copying.
    SIZE_T totalSent = 0;               // # of bytes totally received.
    SIZE_T lastSent = 0;                // # of bytes received in last op.

    do {
        lastSent = (cntBytes < this->cntBufSend) 
            ? cntBytes : this->cntBufSend;

        ::memcpy(this->bufSend, in, lastSent);

        result = ::rdma_post_send(this->id, NULL, this->bufSend, 
            lastSent, this->mrSend, 0);
        if (result != 0) {
            throw IbRdmaException("rdma_post_send", errno, __FILE__, __LINE__);
        }

        //result = ::rdma_get_send_comp(this->id, &wc);
        //if (result != 0) {
        //    throw IbRdmaException("rdma_get_send_comp", errno, __FILE__, __LINE__);
        //}
        while (!::ibv_poll_cq(this->id->send_cq, 1, &wc));

        totalSent += lastSent;
        in += lastSent;

    } while (forceSend && (totalSent < cntBytes));

    return totalSent;
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::IbRdmaCommClientChannel
 */
vislib::net::ib::IbRdmaCommClientChannel::IbRdmaCommClientChannel(void) 
        : bufRecv(NULL), bufRecvEnd(NULL), bufSend(NULL), bufSendEnd(NULL), 
        cntBufRecv(0), cntBufSend(0), id(NULL), mrRecv(NULL), mrSend(NULL) {
    VLSTACKTRACE("IbRdmaCommClientChannel::IbRdmaCommClientChannel", 
        __FILE__, __LINE__);
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::~IbRdmaCommClientChannel
 */
vislib::net::ib::IbRdmaCommClientChannel::~IbRdmaCommClientChannel(void) {
    VLSTACKTRACE("IbRdmaCommClientChannel::~IbRdmaCommClientChannel", 
        __FILE__, __LINE__);
    try {
        this->Close();
    } catch (...) {
        VLTRACE(Trace::LEVEL_VL_WARN, "An exception was caught when closing "
            "a IbRdmaCommClientChannel in its destructor.\n");
    }

    this->setBuffers(NULL, 0, NULL, 0);
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::postReceive
 */
void vislib::net::ib::IbRdmaCommClientChannel::postReceive(void) {
    VLSTACKTRACE("IbRdmaCommClientChannel::postReceive", __FILE__, __LINE__);
    int result = ::rdma_post_recv(this->id, NULL, this->bufRecv, 
        this->cntBufRecv, this->mrRecv);
    if (result != 0) {
        throw IbRdmaException("rdma_post_recv", errno, __FILE__, __LINE__);
    }
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::registerBuffers
 */
void vislib::net::ib::IbRdmaCommClientChannel::registerBuffers(void) {
    VLSTACKTRACE("IbRdmaCommClientChannel::registerBuffers", __FILE__, 
        __LINE__);
    ASSERT(this->id != NULL);
    ASSERT(this->mrRecv == NULL);
    ASSERT(this->bufRecv != NULL);
    ASSERT(this->mrSend == NULL);
    ASSERT(this->bufSend != NULL);

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
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::setBuffers
 */
void vislib::net::ib::IbRdmaCommClientChannel::setBuffers(BYTE *bufRecv, 
        const SIZE_T cntBufRecv, BYTE *bufSend, const SIZE_T cntBufSend) {
    VLSTACKTRACE("IbRdmaCommClientChannel::setBuffers", __FILE__, __LINE__);
    
    /* If there is a current buffer and if it is owned by us, delete it. */
    if (this->bufRecvEnd == NULL) {
        // Note: This is not wrong! 'bufRecvEnd' is indicator for ownership of
        // 'bufRecv'.
        ARY_SAFE_DELETE(this->bufRecv);

    } else {
        this->bufRecvEnd = NULL;
    }

    if (this->bufSendEnd == NULL) {
        // Note: This is not wrong! Same as above...
        ARY_SAFE_DELETE(this->bufSend);

    } else {
        this->bufRecvEnd = NULL;
    }

    ASSERT(this->bufRecv == NULL);
    ASSERT(this->bufRecvEnd == NULL);
    ASSERT(this->bufSend == NULL);
    ASSERT(this->bufSendEnd == NULL);

    /* Remember the buffer size. */
    this->cntBufRecv = cntBufRecv;
    this->cntBufSend = cntBufSend;

    /* Allocate own buffers or set user buffers including indicator. */
    if (this->cntBufRecv > 0) {
        if (bufRecv == NULL) {
            VLTRACE(Trace::LEVEL_VL_VERBOSE, "Allocating RDMA receive "
                "buffer of %d bytes...\n", this->cntBufRecv);
            this->bufRecv = new BYTE[this->cntBufRecv];
            this->bufRecvEnd = NULL;

        } else {
            VLTRACE(Trace::LEVEL_VL_VERBOSE, "Using user-specified RDMA "
                "receive buffer.\n");
            this->bufRecv = bufRecv;
            this->bufRecvEnd = this->bufRecv + this->cntBufRecv;
        }
    }

    if (this->cntBufSend > 0) {
        if (bufSend == NULL) {
            VLTRACE(Trace::LEVEL_VL_VERBOSE, "Allocating RDMA send "
                "buffer of %d bytes...\n", this->cntBufRecv);
            this->bufSend = new BYTE[this->cntBufSend];
            this->bufSendEnd = NULL;

        } else {
            VLTRACE(Trace::LEVEL_VL_VERBOSE, "Using user-specified RDMA "
                "send buffer.\n");
            this->bufSend = bufSend;
            this->bufSendEnd = this->bufSend + this->cntBufSend;
        }
    }
}

