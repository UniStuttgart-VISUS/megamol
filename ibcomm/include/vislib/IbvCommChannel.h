/*
 * IbvCommChannel.h
 *
 * Copyright (C) 2006 - 2011 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_IBVCOMMCHANNEL_H_INCLUDED
#define VISLIB_IBVCOMMCHANNEL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Socket.h"                      // Must be first!
#include "vislib/AbstractCommServerChannel.h"

#include "rdma/rdma_cma.h"
#include "rdma/winverbs.h"


namespace vislib {
namespace net {
namespace ib {


    /**
     * TODO: comment class
     */
    class IbvCommChannel : AbstractCommServerChannel {

    public:

        /**
         * Permit incoming connection attempt on the communication channel.
         *
         * @return The client connection.
         *
         * @throws SocketException In case the operation fails.
         */
        virtual SmartRef<AbstractCommChannel> Accept(void);

        /**
         * Binds the server to a specified end point address.
         *
         * @param endPoint The end point address to bind to.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual void Bind(SmartRef<AbstractCommEndPoint> endPoint);

        /**
         * Place the communication channel in a state in which it is listening
         * for an incoming connection.
         *
         * @param backlog Maximum length of the queue of pending connections.
         *
         * @throws SocketException In case the operation fails.
         */
        virtual void Listen(const int backlog = SOMAXCONN);

    protected:

        /** Superclass typedef. */
        typedef AbstractCommServerChannel Super;

        /** Ctor. */
        IbvCommChannel(void);

        /** Dtor. */
        virtual ~IbvCommChannel(void);

    private:

        IWVConnectEndpoint *connectEndPoint;

        IWVProvider *wvProvider;

    };
    
} /* end namespace ib */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_IBVCOMMCHANNEL_H_INCLUDED */

