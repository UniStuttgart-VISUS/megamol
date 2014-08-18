///*
// * IbvCommChannel.h
// *
// * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart. 
// * Alle Rechte vorbehalten.
// */
//
//#ifndef VISLIB_IBVCOMMCHANNEL_H_INCLUDED
//#define VISLIB_IBVCOMMCHANNEL_H_INCLUDED
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
//    class IbvCommChannel : public AbstractCommServerChannel {
//
//    public:
//
//        /**
//         * Create a new channel.
//         *
//         * @param flags The flags for the channel.
//         */
//        static inline SmartRef<IbvCommChannel> Create(void) {
//            return SmartRef<IbvCommChannel>(new IbvCommChannel(), false);
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
//        /**
//         * Answer the WinVerbs provider used by the channel. 
//         *
//         * You should call AddRef() - and later Release() - if you intend to 
//         * store this instance in order to ensure a correct life time handling
//         * of the object.
//         *
//         * @return The WinVerbs provides used by the channel.
//         */
//        IWVProvider *GetProvider(void);
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
//        /** Ctor. */
//        IbvCommChannel(IWVProvider *wvProvider = NULL);
//
//        /** Dtor. */
//        virtual ~IbvCommChannel(void);
//
//        void createQueuePair(void);
//
//        void initialise(void);
//
//        void releaseQueuePair(void);
//
//    private:
//
//        IWVConnectEndpoint *connectEndPoint;
//
//        IWVDevice *device;
//
//        IWVProtectionDomain *protectionDomain;
//
//        IWVConnectQueuePair *queuePair;
//
//        IWVCompletionQueue *recvComplQueue;
//
//        WV_MEMORY_KEYS recvKeys;
//
//        void *recvRegion;
//
//        /** Size of 'recvRegion' in bytes. */
//        SIZE_T recvRegionSize;
//
//        IWVCompletionQueue *sendComplQueue;
//
//        WV_MEMORY_KEYS sendKeys;
//
//        void *sendRegion;
//
//        SIZE_T sendRegionSize;
//
//        IWVProvider *wvProvider;
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
//#endif /* VISLIB_IBVCOMMCHANNEL_H_INCLUDED */
//
