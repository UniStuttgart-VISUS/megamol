/*
 * IbRdmaCommServerChannel.h
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_IBRDMACOMMSERVERCHANNEL_H_INCLUDED
#define VISLIB_IBRDMACOMMSERVERCHANNEL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Socket.h"                      // Must be first!
#include "vislib/AbstractCommServerChannel.h"
#include "vislib/IbRdmaCommClientChannel.h"
#include "vislib/IbRdmaException.h"
#include "vislib/StackTrace.h"

#include "rdma/rdma_cma.h"
#include "rdma/rdma_verbs.h"


namespace vislib {
namespace net {
namespace ib {


    /**
     * TODO: comment class
     */
    class IbRdmaCommServerChannel : public AbstractCommServerChannel {

    public:

        static SmartRef<IbRdmaCommServerChannel> Create(const SIZE_T cntBufRecv,
            const SIZE_T cntBufSend);

        static SmartRef<IbRdmaCommServerChannel> Create(const SIZE_T cntBuf);

        virtual SmartRef<AbstractCommClientChannel> Accept(void);

        SmartRef<IbRdmaCommClientChannel> Accept(
                BYTE *bufRecv, const SIZE_T cntBufRecv, 
                BYTE *bufSend, const SIZE_T cntBufSend);

        virtual void Bind(SmartRef<AbstractCommEndPoint> endPoint);

        virtual void Close(void);

        virtual SmartRef<AbstractCommEndPoint> GetLocalEndPoint(void) const;

        virtual void Listen(const int backlog = SOMAXCONN);

    private:

        /** Superclass typedef. */
        typedef AbstractCommServerChannel Super;

        /** Ctor. */
        IbRdmaCommServerChannel(const SIZE_T cntBufRecv, 
            const SIZE_T cntBufSend);

        /** Dtor. */
        ~IbRdmaCommServerChannel(void);

        /** Size of receive buffers created for client channels in bytes. */
        SIZE_T cntBufRecv;

        /** Size of send buffers created for client channels in bytes. */
        SIZE_T cntBufSend;

        /** The root handle of the RMDA objects used by this channel. */
        struct rdma_cm_id *id;

        /** 
         * Stores the initialisation parameters for a queue pair. We need this
         * member for a hack implemented in IbRdmaCommServerChannel::Accept().
         */
        struct ibv_qp_init_attr qpAttr;

    };
    
} /* end namespace ib */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_IBRDMACOMMSERVERCHANNEL_H_INCLUDED */

