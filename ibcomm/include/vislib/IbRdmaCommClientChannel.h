/*
 * IbRdmaCommClientChannel.h
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_IBRDMACOMMCLIENTCHANNEL_H_INCLUDED
#define VISLIB_IBRDMACOMMCLIENTCHANNEL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Socket.h"                      // Must be first!
#include "vislib/AbstractCommClientChannel.h"
#include "vislib/IbRdmaException.h"

#include "rdma/rdma_cma.h"
#include "rdma/rdma_verbs.h"



namespace vislib {
namespace net {
namespace ib {


    /**
     * TODO: comment class
     */
    class IbRdmaCommClientChannel : public AbstractCommClientChannel {

    public:

        /**
         * Create a new instance with receive and send buffers of the specified
         * size. The buffers will be allocated by the object.
         *
         * @param cntBufRecv The size of the receive buffer in bytes.
         * @param cntBufSend The size of the send buffer in bytes.
         *
         * @return
         */
        static SmartRef<IbRdmaCommClientChannel> Create(
            const SIZE_T cntBufRecv, const SIZE_T cntBufSend);

        static SmartRef<IbRdmaCommClientChannel> Create(const SIZE_T cntBuf);

        virtual void Close(void);

        virtual void Connect(SmartRef<AbstractCommEndPoint> endPoint);

        virtual SmartRef<AbstractCommEndPoint> GetLocalEndPoint(void) const;

        virtual SmartRef<AbstractCommEndPoint> GetRemoteEndPoint(void) const;

        virtual SIZE_T Receive(void *outData, const SIZE_T cntBytes,
            const UINT timeout = TIMEOUT_INFINITE, 
            const bool forceReceive = true);

        virtual SIZE_T Send(const void *data, const SIZE_T cntBytes,
            const UINT timeout = TIMEOUT_INFINITE, 
            const bool forceSend = true);

    private:

        /** Superclass typedef. */
        typedef AbstractCommClientChannel Super;

        /** Ctor. */
        IbRdmaCommClientChannel(void);

        /** Dtor. */
        ~IbRdmaCommClientChannel(void);

        void setBuffers(BYTE *bufRecv, const SIZE_T cntBufRecv, 
            BYTE *bufSend, const SIZE_T cntBufSend);

        /**
         * Buffer for receiving data from the network. This can either be a 
         * buffer owned by the channel or a user-provided memory range (for
         * zero-copy operations). This pointer is registered via 'mrRecv'.
         */
        BYTE *bufRecv;

        BYTE *bufRecvEnd;

        /**
         * Buffer for sending data from the network. This can either be a 
         * buffer owned by the channel or a user-provided memory range (for
         * zero-copy operations). This pointer is registered via 'mrSend'.
         */
        BYTE *bufSend;

        BYTE *bufSendEnd;

        /** Size of 'bufRecv' in bytes. */
        SIZE_T cntBufRecv;

        /** Size of 'bufSend' in bytes. */
        SIZE_T cntBufSend;

        struct rdma_cm_id *id;

        struct ibv_mr *mrRecv;

        struct ibv_mr *mrSend;

        /** 
         * The server channel must be able to initialise a client channel 
         * manually in the Accept() method. Therefore, it must be a friend.
         */
        friend class IbRdmaCommServerChannel;

    };
    
} /* end namespace ib */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_IBRDMACOMMCLIENTCHANNEL_H_INCLUDED */

