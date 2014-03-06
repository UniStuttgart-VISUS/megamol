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
#include "the/memory.h"
#include "the/trace.h"


/*
 * vislib::net::ib::IbRdmaCommClientChannel::Create
 */
vislib::SmartRef<vislib::net::ib::IbRdmaCommClientChannel> 
vislib::net::ib::IbRdmaCommClientChannel::Create(const size_t cntBufRecv, 
        const size_t cntBufSend) {
    THE_STACK_TRACE;
    return IbRdmaCommClientChannel::Create(NULL, cntBufRecv, NULL, 
        cntBufSend);
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::Create
 */
vislib::SmartRef<vislib::net::ib::IbRdmaCommClientChannel> 
vislib::net::ib::IbRdmaCommClientChannel::Create(const size_t cntBuf) {
    THE_STACK_TRACE;
    return IbRdmaCommClientChannel::Create(NULL, cntBuf, NULL, cntBuf);
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::Create
 */
vislib::SmartRef<vislib::net::ib::IbRdmaCommClientChannel> 
vislib::net::ib::IbRdmaCommClientChannel::Create(uint8_t *bufRecv, 
        const size_t cntBufRecv, uint8_t *bufSend, const size_t cntBufSend) {
    THE_STACK_TRACE;
    SmartRef<IbRdmaCommClientChannel> retval(new IbRdmaCommClientChannel(), 
        false);
    retval->setBuffers(bufRecv, cntBufRecv, bufSend, cntBufSend);
    return retval;
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::Close
 */
void vislib::net::ib::IbRdmaCommClientChannel::Close(void) {
    THE_STACK_TRACE;

    int result = 0;                     // RDMA API results.

    result = ::rdma_disconnect(this->id);
    if (result != 0) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "rdma_disconnect failed with error "
            "code %d when closing the channel.\n", errno);
    }

    if (this->mrRecv != NULL) {
        ::rdma_dereg_mr(this->mrRecv);
        this->mrRecv = NULL;
    }

    if (this->mrSend != NULL) {
        ::rdma_dereg_mr(this->mrSend);
        this->mrSend = NULL;
    }

    if (this->id != NULL) {
        ::rdma_destroy_ep(this->id);
        this->id = NULL;
    }
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::Connect
 */
void vislib::net::ib::IbRdmaCommClientChannel::Connect(
        SmartRef<AbstractCommEndPoint> endPoint) {
    THE_STACK_TRACE;

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
    attr.cap.max_inline_data = 0;  // TODO
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
    THE_STACK_TRACE;
    WV_CONNECT_ATTRIBUTES attribs;
    this->id->ep.connect->Query(&attribs);
    return IPCommEndPoint::Create(attribs.LocalAddress.Sin);
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::GetRemoteEndPoint
 */
vislib::SmartRef<vislib::net::AbstractCommEndPoint>
vislib::net::ib::IbRdmaCommClientChannel::GetRemoteEndPoint(void) const {
    THE_STACK_TRACE;
    WV_CONNECT_ATTRIBUTES attribs;
    this->id->ep.connect->Query(&attribs);
    return IPCommEndPoint::Create(attribs.PeerAddress.Sin);
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::Receive
 */
size_t vislib::net::ib::IbRdmaCommClientChannel::Receive(void *outData, 
        const size_t cntBytes, const unsigned int timeout, const bool forceReceive) {
    THE_STACK_TRACE;

    int result = 0;                     // RDMA API results.
    struct ibv_wc wc;                   // Receives the completion parameters.
    uint8_t *outPtr = static_cast<uint8_t *>(outData);    // Cursor through 'outData'.
    size_t lastReceived = 0;            // # of bytes received in last op.
    size_t totalReceived = 0;           // # of bytes totally received.
    

    if (this->IsZeroCopyReceive()) {
        // The user has specified the DMA memory area directly, so complete
        // the in-flight receive and post the next one. 'forceReceive' is not
        // allowed in this operation mode.

        THE_ASSERT(false);  // TODO: This currently does not work.

        //THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Waiting for RDMA receive "
        //    "operation to complete...\n");
        //while (!::ibv_poll_cq(this->id->recv_cq, 1, &wc));

        //// TODO: Currently, we can only receive to the begin of the buffer.
        //THE_ASSERT((outData == NULL) || (outData == this->bufRecv));
        ////outPtr = this->bufRecv;

        //if (cntBytes > this->cntBufRecv) {
        //    // User tries to receive more than the supplied memory range.
        //    throw IllegalParamException("cntBytes", __FILE__, __LINE__);
        //}

        //this->postReceive();

        //totalReceived = cntBytes;

    } else {
        do {
            if (this->cntRemRecv > 0) {
                THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Using %d bytes of "
                    "RDMA data left from last receive operation.\n", cntRemRecv);
                if (cntBytes > this->cntRemRecv) {
                    lastReceived = this->cntRemRecv;
                } else {
                    lastReceived = cntBytes;
                }

                // Copy receive data to user buffer.
                THE_ASSERT(lastReceived <= this->cntRemRecv);
                ::memcpy(outPtr, this->remRecv, lastReceived);

                // Update internal cursor variables.
                this->remRecv += lastReceived;
                this->cntRemRecv -= lastReceived;

                // Update current call cursor variables.
                totalReceived += lastReceived;
                outPtr += lastReceived;

            } else {

                // Complete the current in-flight receive operation. This will block
                // until the data becomes available.
                THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Waiting for RDMA "
                    "receive operation to complete...\n");
                do {
                    if (this->id == NULL) {
                        throw IbRdmaException(EOWNERDEAD, __FILE__, __LINE__);
                    }
                } while (!::ibv_poll_cq(this->id->recv_cq, 1, &wc));
                THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Received %d bytes "
                    "via RDMA, status %d.\n", wc.byte_len, wc.status);


                // TODO: The following code does not work. The hack above was found at
                // http://www.spinics.net/lists/linux-rdma/msg04795.html
                //result = ::rdma_get_recv_comp(this->id, &wc);
                //if (result != 0) {
                //    throw IbRdmaException("rdma_get_recv_comp", errno, __FILE__, __LINE__);
                //}

                // Re-initialise internal buffer cursors.
                this->remRecv = this->bufRecv;
                this->cntRemRecv = wc.byte_len;
                THE_ASSERT(this->cntRemRecv <= this->cntBufRecv);

                // Determine how much we can copy from the receive buffer to the 
                // user-supplied destination buffer.
                lastReceived = cntBytes - totalReceived;
                if (lastReceived > this->cntRemRecv) {
                    // User wants too much...
                    lastReceived = this->cntRemRecv;
                }

                // Copy receive data to user buffer.
                THE_ASSERT(lastReceived <= this->cntRemRecv);
                THE_ASSERT(this->remRecv == this->bufRecv);
                ::memcpy(outPtr, this->remRecv, lastReceived);

                // Update internal cursor variables.
                this->remRecv += lastReceived;
                this->cntRemRecv -= lastReceived;

                // Update current call cursor variables.
                totalReceived += lastReceived;
                outPtr += lastReceived;
            }
            THE_ASSERT(this->cntRemRecv >= 0);
            THE_ASSERT(this->cntRemRecv < this->cntBufRecv);

            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "User did not receive "
                "%u bytes already in RDMA buffer.\n", this->cntRemRecv);

            if (this->cntRemRecv == 0) {
                // Post receive for next iteration or next call to Receive().
                this->postReceive();
            }

        } while (forceReceive && (totalReceived < cntBytes) && (lastReceived > 0));
    }

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
size_t vislib::net::ib::IbRdmaCommClientChannel::Send(const void *data, 
        const size_t cntBytes, const unsigned int timeout,  const bool forceSend) {
    THE_STACK_TRACE;

    int result = 0;                     // RDMA API results.
    struct ibv_wc wc;                   // Receives the completion parameters.
    const uint8_t *inPtr = static_cast<const uint8_t *>(data);    // Cursor in 'data'.
    size_t totalSent = 0;               // # of bytes totally received.
    size_t lastSent = 0;                // # of bytes received in last op.

    if (this->IsZeroCopySend()) {
        THE_ASSERT(this->bufSendEnd != NULL);

        /* 
         * If no specific location to send from is specified, assume the begin
         * of the registered DMA memory and fix the cursor automagically.
         */
        if (inPtr == NULL) {
            inPtr = this->bufSend;
        }

        /* 
         * The send from user-specified DMA memory must succeed as a whole, 
         * i. e. the data cannot be sent incrementally. In turn, 'totalSent'
         * (the method return value) and 'cntBytes' must be the same.
         */
        totalSent = cntBytes;

        /* Sanity check for data range being sent. */
        if ((inPtr + totalSent) > this->bufSendEnd) {
            throw IllegalParamException("cntBytes", __FILE__, __LINE__);
        }

        /* Send the data. */
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Posting RDMA send "
            "of %d bytes starting at %p...\n", totalSent, inPtr);
        result = ::rdma_post_send(this->id, NULL, const_cast<uint8_t *>(inPtr), 
            totalSent, this->mrSend, 0);
        if (result != 0) {
            throw IbRdmaException("rdma_post_send", errno, __FILE__, __LINE__);
        }

        /* Wait for completion of send operation. */
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Waiting for RDMA send "
            "operation to complete...\n");
        while (!::ibv_poll_cq(this->id->send_cq, 1, &wc));
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "RDMA send operation "
            "completed.\n");

    } else {
        do {
            THE_ASSERT(cntBytes >= totalSent);
            if ((cntBytes - totalSent) < this->cntBufSend) {
                lastSent = cntBytes - totalSent;
            } else {
                lastSent = this->cntBufSend;
            }

            ::memcpy(this->bufSend, inPtr, lastSent);

            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Posting RDMA send "
                "of %u bytes (buffer %u bytes) starting at %p...\n", lastSent, 
                this->cntBufSend, this->bufSend);
            if (this->id == NULL) {
                throw IbRdmaException(EOWNERDEAD, __FILE__, __LINE__);
            }
            result = ::rdma_post_send(this->id, NULL, this->bufSend, 
                lastSent, this->mrSend, 0);
            if (result != 0) {
                throw IbRdmaException("rdma_post_send", errno, __FILE__, __LINE__);
            }

            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Waiting for RDMA send "
                "operation to complete...\n");
            do {
                if (this->id == NULL) {
                    throw IbRdmaException(EOWNERDEAD, __FILE__, __LINE__);
                }
            } while (!::ibv_poll_cq(this->id->send_cq, 1, &wc));
            // TODO: The following does not work for some reason (same as for Receive()):
            //result = ::rdma_get_send_comp(this->id, &wc);
            //if (result != 0) {
            //    throw IbRdmaException("rdma_get_send_comp", errno, __FILE__, __LINE__);
            //}
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "RDMA send operation "
                "completed for %u bytes, status %d.\n", wc.byte_len, wc.status);

            totalSent += lastSent;
            inPtr += lastSent;


        } while (forceSend && (totalSent < cntBytes));
    }

    return totalSent;
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::IbRdmaCommClientChannel
 */
vislib::net::ib::IbRdmaCommClientChannel::IbRdmaCommClientChannel(void) 
        : bufRecv(NULL), bufRecvEnd(NULL), bufSend(NULL), bufSendEnd(NULL), 
        cntBufRecv(0), cntBufSend(0), cntRemRecv(0), id(NULL), 
        mrRecv(NULL), mrSend(NULL), remRecv(NULL) {
    THE_STACK_TRACE;
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::~IbRdmaCommClientChannel
 */
vislib::net::ib::IbRdmaCommClientChannel::~IbRdmaCommClientChannel(void) {
    THE_STACK_TRACE;
    //try {
    //    this->Close();
    //} catch (...) {
    //    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "An exception was caught when closing "
    //        "a IbRdmaCommClientChannel in its destructor.\n");
    //}

    this->setBuffers(NULL, 0, NULL, 0);
}


/*
 * vislib::net::ib::IbRdmaCommClientChannel::postReceive
 */
void vislib::net::ib::IbRdmaCommClientChannel::postReceive(void) {
    THE_STACK_TRACE;
    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Posting RDMA receive of %d "
        "bytes into %p...\n", this->cntBufRecv, this->bufRecv);
    if (this->id == NULL) {
        throw IbRdmaException(EOWNERDEAD, __FILE__, __LINE__);
    }
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
    THE_STACK_TRACE;
    THE_ASSERT(this->id != NULL);
    THE_ASSERT(this->mrRecv == NULL);
    THE_ASSERT(this->bufRecv != NULL);
    THE_ASSERT(this->mrSend == NULL);
    THE_ASSERT(this->bufSend != NULL);

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
void vislib::net::ib::IbRdmaCommClientChannel::setBuffers(uint8_t *bufRecv, 
        const size_t cntBufRecv, uint8_t *bufSend, const size_t cntBufSend) {
    THE_STACK_TRACE;
    
    /* If there is a current buffer and if it is owned by us, delete it. */
    if (!this->IsZeroCopyReceive()) {
        the::safe_array_delete(this->bufRecv);
    } else {
        this->bufRecvEnd = NULL;
    }

    if (!this->IsZeroCopySend()) {
        the::safe_array_delete(this->bufSend);
    } else {
        this->bufRecvEnd = NULL;
    }

    THE_ASSERT(this->bufRecv == NULL);
    THE_ASSERT(this->bufRecvEnd == NULL);
    THE_ASSERT(this->bufSend == NULL);
    THE_ASSERT(this->bufSendEnd == NULL);

    /* Remember the buffer size. */
    this->cntBufRecv = cntBufRecv;
    this->cntBufSend = cntBufSend;

    /* Allocate own buffers or set user buffers including indicator. */
    if (this->cntBufRecv > 0) {
        if (bufRecv == NULL) {
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Allocating RDMA receive "
                "buffer of %d bytes...\n", this->cntBufRecv);
            this->bufRecv = new uint8_t[this->cntBufRecv]; 
            this->bufRecvEnd = NULL;

        } else {
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Using user-specified RDMA "
                "receive buffer.\n");
            this->bufRecv = bufRecv;
            this->bufRecvEnd = this->bufRecv + this->cntBufRecv;
        }
    }

    if (this->cntBufSend > 0) {
        if (bufSend == NULL) {
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Allocating RDMA send "
                "buffer of %d bytes...\n", this->cntBufRecv);
            this->bufSend = new uint8_t[this->cntBufSend];
            this->bufSendEnd = NULL;

        } else {
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "Using user-specified RDMA "
                "send buffer.\n");
            this->bufSend = bufSend;
            this->bufSendEnd = this->bufSend + this->cntBufSend;
        }
    }
}

