///*
// * IbvCommServerChannel.h
// *
// * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart. 
// * Alle Rechte vorbehalten.
// */
//
//#ifndef VISLIB_IBVCOMMSERVERCHANNEL_H_INCLUDED
//#define VISLIB_IBVCOMMSERVERCHANNEL_H_INCLUDED
//#if (defined(_MSC_VER) && (_MSC_VER > 1000))
//#pragma once
//#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
//#if defined(_WIN32) && defined(_MANAGED)
//#pragma managed(push, off)
//#endif /* defined(_WIN32) && defined(_MANAGED) */
//
//
//#include "vislib/Socket.h"                      // Must be first!
//#include "vislib/AbstractCommServerChannel.h"
//#include "vislib/COMException.h"
//#include "vislib/Event.h"
//#include "vislib/StackTrace.h"
//#include "vislib/Thread.h"
//
//#include "rdma/rdma_cma.h"
//#include "rdma/winverbs.h"
//
//
//namespace vislib {
//namespace net {
//namespace ib {
//
//
//    /**
//     * TODO: comment class
//     */
//    class IbvCommServerChannel : public AbstractCommServerChannel {
//
//    public:
//
//        static inline SmartRef<IbvCommServerChannel> Create(void) {
//            VLSTACKTRACE("IbvCommServerChannel::Create", __FILE__, __LINE__);
//            return SmartRef<IbvCommServerChannel>(new IbvCommServerChannel(), 
//                false);
//        }
//
//        virtual SmartRef<AbstractCommChannel> Accept(void);
//
//        virtual void Bind(SmartRef<AbstractCommEndPoint> endPoint);
//
//        virtual void Close(void);
//
//        virtual void Connect(SmartRef<AbstractCommEndPoint> endPoint);
//
//        virtual SmartRef<AbstractCommEndPoint> GetLocalEndPoint(void) const;
//
//        virtual SmartRef<AbstractCommEndPoint> GetRemoteEndPoint(void) const;
//
//        virtual void Listen(const int backlog = SOMAXCONN);
//
//        virtual SIZE_T Receive(void *outData, const SIZE_T cntBytes,
//            const UINT timeout = TIMEOUT_INFINITE, 
//            const bool forceReceive = true);
//
//        virtual SIZE_T Send(const void *data, const SIZE_T cntBytes,
//            const UINT timeout = TIMEOUT_INFINITE, 
//            const bool forceSend = true);
//
//    protected:
//
//        /** Superclass typedef. */
//        typedef AbstractCommServerChannel Super;
//
//    private:
//
//        static DWORD messagePump(void *userData); 
//
//        /** Ctor. */
//        IbvCommServerChannel(void);
//
//        /** Dtor. */
//        virtual ~IbvCommServerChannel(void);
//
//        int onConnectRequest(struct rdma_cm_event& evt);
//        
//        struct rdma_event_channel *channel;
//        struct rdma_cm_event *evt;
//        struct rdma_cm_id *id;
//
//        sys::Event evtAccept;
//        sys::Thread msgThread;
//
//
//    };
//    
//} /* end namespace ib */
//} /* end namespace net */
//} /* end namespace vislib */
//
//#if defined(_WIN32) && defined(_MANAGED)
//#pragma managed(pop)
//#endif /* defined(_WIN32) && defined(_MANAGED) */
//#endif /* VISLIB_IBVCOMMSERVERCHANNEL_H_INCLUDED */
//
