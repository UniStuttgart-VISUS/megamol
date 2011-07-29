/*
 * HeartbeatClient.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTER_SIMPLE_HEARTBEATCLIENT_H_INCLUDED
#define MEGAMOLCORE_CLUSTER_SIMPLE_HEARTBEATCLIENT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/RawStorage.h"
#include "vislib/SmartRef.h"
#include "vislib/String.h"
#include "vislib/TcpCommChannel.h"
#include "vislib/Thread.h"


namespace megamol {
namespace core {
namespace cluster {
namespace simple {


    /**
     * Abstract base class of override rendering views
     */
    class HeartbeatClient {
    public:

        /** Ctor. */
        HeartbeatClient(void);

        /** Dtor. */
        virtual ~HeartbeatClient(void);

        /**
         * Connects this client to the specified server.
         * If the client is already connected, this connection is closed
         *
         * @param server The server of the heartbeat to connect to
         * @param port The port of the heartbeat to connect to
         */
        void Connect(vislib::StringW server, unsigned int port);

        /**
         * Closes the connection and shuts the client down.
         * It is save to shut a client down which is not connected
         */
        void Shutdown(void);

        /**
         * Synchronises to the heartbeat
         *
         * @param outPayload The data received from the heartbeat server as payload
         *
         * @return True if the client is connected and the payload is valid.
         *         False if the client is not connected. The payload should be ignored then.
         */
        bool Sync(vislib::RawStorage& outPayload);

    private:

        /**
         * Connection thread
         *
         * @param userData Pointer to this object
         *
         * @return 0
         */
        static DWORD connector(void *userData);

        /** The communication channel */
        vislib::SmartRef<vislib::net::TcpCommChannel> chan;

        /** The connection thread */
        vislib::sys::Thread conn;

        /* The port of the heartbeat to connect to */
        unsigned int port;

        /* The server of the heartbeat to connect to */
        vislib::StringW server;

    };


} /* end namespace simple */
} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTER_SIMPLE_HEARTBEATCLIENT_H_INCLUDED */
