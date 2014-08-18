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
 #include "vislib/StackTrace.h"

#include "rdma/rdma_cma.h"
#include "rdma/rdma_verbs.h"



namespace vislib {
namespace net {
namespace ib {


    /**
     * This class implements the client channel interface of the VISlib comm 
     * channel infrastructure using the RDMA API for InfiniBand.
     *
     * Objects of this class should either be used for establishing a connection
     * to an InfiniBand RDMA server or are returned by the InfiniBand RDMA
     * server in its Accept() method.
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
        
        /**
         * Create a new instance with receive and send buffers of the specified
         * size. The buffers will be allocated by the object.
         *
         * @param cntBuf The size of the receive and send buffer in bytes.
         *
         * @return
         */
        static SmartRef<IbRdmaCommClientChannel> Create(const SIZE_T cntBuf);

        /**
         * Create a new instance using the specified buffers for DMA.
         *
         * @param bufRecv    A pointer to the receive buffer. If NULL, the 
         *                   method will allocate 'cntBufRecv' bytes for the
         *                   buffer. Otherwise, the memory provided will be 
         *                   used. In this case, the caller remains owner of
         *                   the memory and must ensure that it is available
         *                   as long as the new object exists.
         * @param cntBufRecv The size of 'bufRecv' in bytes if specified or the
         *                   number of bytes to be allocaated if 'bufRecv' is
         *                   NULL.
         * @param bufSend    A pointer to the send buffer. If NULL, the 
         *                   method will allocate 'cntBufSend' bytes for the
         *                   buffer. Otherwise, the memory provided will be 
         *                   used. In this case, the caller remains owner of
         *                   the memory and must ensure that it is available
         *                   as long as the new object exists.
         * @param cntBufSend The size of 'bufSend' in bytes if specified or the
         *                   number of bytes to be allocaated if 'bufSend' is
         *                   NULL.
         *
         * @return
         */
        static SmartRef<IbRdmaCommClientChannel> Create(BYTE *bufRecv, 
            const SIZE_T cntBufRecv, BYTE *bufSend, const SIZE_T cntBufSend);

        virtual void Close(void);

        virtual void Connect(SmartRef<AbstractCommEndPoint> endPoint);

        virtual SmartRef<AbstractCommEndPoint> GetLocalEndPoint(void) const;

        virtual SmartRef<AbstractCommEndPoint> GetRemoteEndPoint(void) const;

        /**
         * Answer whether the channel directly receives into a user-supplied
         * memory area.
         *
         * @return
         */
        inline bool IsZeroCopyReceive(void) const {
            VLSTACKTRACE("IbRdmaCommClientChannel::IsZeroCopyReceive",
                __FILE__, __LINE__);
            return (this->bufRecvEnd != NULL);
        }

        /**
         * Answer whether the channel directly sends from a user-supplied
         * memory area.
         *
         * @return
         */
        inline bool IsZeroCopySend(void) const {
            VLSTACKTRACE("IbRdmaCommClientChannel::IsZeroCopySend", 
                __FILE__, __LINE__);
            return (this->bufSendEnd != NULL);
        }

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

        /**
         * Post a receive of the full receive buffer size to the completion
         * queue.
         *
         * @throws IbRdmaException If the operation failed.
         */
        void postReceive(void);

        /**
         * Register the receive and the send buffer with the RMDA ID. It is
         * required that the RDMA ID and all buffers have been allocated before.
         * No buffers must have been registered before the method is called.
         */
        void registerBuffers(void);

        /**
         * Changes the send and receive buffers.
         *
         * @param bufRecv    A pointer to the receive buffer. If NULL, the 
         *                   method will allocate 'cntBufRecv' bytes for the
         *                   buffer. Otherwise, the memory provided will be 
         *                   used. In this case, the caller remains owner of
         *                   the memory and must ensure that it is available
         *                   as long as the new object exists.
         * @param cntBufRecv The size of 'bufRecv' in bytes if specified or the
         *                   number of bytes to be allocaated if 'bufRecv' is
         *                   NULL.
         * @param bufSend    A pointer to the send buffer. If NULL, the 
         *                   method will allocate 'cntBufSend' bytes for the
         *                   buffer. Otherwise, the memory provided will be 
         *                   used. In this case, the caller remains owner of
         *                   the memory and must ensure that it is available
         *                   as long as the new object exists.
         * @param cntBufSend The size of 'bufSend' in bytes if specified or the
         *                   number of bytes to be allocaated if 'bufSend' is
         *                   NULL.
         */
        void setBuffers(BYTE *bufRecv, const SIZE_T cntBufRecv, 
            BYTE *bufSend, const SIZE_T cntBufSend);

        /**
         * Buffer for receiving data from the network. This can either be a 
         * buffer owned by the channel or a user-provided memory range (for
         * zero-copy operations). This pointer is registered via 'mrRecv'.
         */
        BYTE *bufRecv;

        /** 
         * Pointer to the end of the receive buffer in case of a zero-copy
         * operation. Otherwise, the pointer is NULL.
         */
        BYTE *bufRecvEnd;

        /**
         * Buffer for sending data from the network. This can either be a 
         * buffer owned by the channel or a user-provided memory range (for
         * zero-copy operations). This pointer is registered via 'mrSend'.
         */
        BYTE *bufSend;

        /** 
         * Pointer to the end of the send buffer in case of a zero-copy
         * operation. Otherwise, the pointer is NULL.
         */
        BYTE *bufSendEnd;

        /** Size of 'bufRecv' in bytes. */
        SIZE_T cntBufRecv;

        /** Size of 'bufSend' in bytes. */
        SIZE_T cntBufSend;

        /** Valid bytes starting at 'remRecv'. */
        SIZE_T cntRemRecv;

        struct rdma_cm_id *id;

        /** Registered memory structure for 'bufRecv'. */
        struct ibv_mr *mrRecv;

        /** Registered memory structure for 'bufSend'. */
        struct ibv_mr *mrSend;

        /** 
         * Pointer to the start of the data in 'bufRecv' that have not yet been
         * delivered to the user.
         */
        BYTE *remRecv;

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
