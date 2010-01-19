/*
 * RenderNetUtil.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_RENDERNETUTIL_H_INCLUDED
#define MEGAMOLCORE_RENDERNETUTIL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "special/RenderNetMsg.h"
#include "vislib/Socket.h"
#include "vislib/String.h"


namespace megamol {
namespace core {
namespace special {

    /**
     * static utility class for network communication within a rendering 
     * network
     */
    class RenderNetUtil {
    public:

        /** The number of message trips to be performed for a time sync */
        static const unsigned int TimeSyncTripCnt = 10;

        /** message data of a time sync message */
        typedef struct _timesyncdata_t {

            /** The number of this trip */
            unsigned int trip;

            /** The times when the messages were sent from the server */
            double srvrTimes[TimeSyncTripCnt];

        } TimeSyncData;

        /** The default network port to be used. */
        static const unsigned short DefaultPort;

        /** Message requesting a time sync */
        static const UINT32 MSGTYPE_REQUEST_TIMESYNC = 0x00000001;

        /** Message performing a time sync */
        static const UINT32 MSGTYPE_TIMESYNC = 0x00000002;

        /** Message requesting a module graph sync */
        static const UINT32 MSGTYPE_REQUEST_MODGRAPHSYNC = 0x00000003;

        /** Message setting up a new module graph */
        static const UINT32 MSGTYPE_SETUP_MODGRAPH = 0x00000004;

        /**
         * Performs the MegaMol™ connection startup handshake as client.
         *
         * @param socket The socket to use
         *
         * @throws vislib::Exception if handshake fails.
         */
        static void HandshakeAsClient(vislib::net::Socket& socket);

        /**
         * Performs the MegaMol™ connection startup handshake as server.
         *
         * @param socket The socket to use
         *
         * @throws vislib::Exception if handshake fails.
         */
        static void HandshakeAsServer(vislib::net::Socket& socket);

        /**
         * Receives a MegaMol™ instance name string
         *
         * @param socket The socket to use
         *
         * @return The returned MegaMol™ instance name string
         */
        static vislib::StringA WhoAreYou(vislib::net::Socket& socket);

        /**
         * Sends a MegaMol™ instance name string
         *
         * @param socket The socket to use
         * @param name The MegaMol™ instance name
         */
        static void ThisIsI(vislib::net::Socket& socket, const vislib::StringA& name);

        /**
         * Generates a name string for this MegaMol™ instance.
         *
         * @return The MegaMol™ instance name
         */
        static vislib::StringA MyName(void);

        /**
         * Receives a message.
         *
         * @param socket The socket to use
         * @param outMsg The variable to receive the message
         */
        static void ReceiveMessage(vislib::net::Socket& socket,
            RenderNetMsg& outMsg);

        /**
         * Sends a message.
         *
         * @param socket The socket to use
         * @param msg The message to be sent
         */
        static void SendMessage(vislib::net::Socket& socket,
            const RenderNetMsg& msg);

    private:

        /** The timeout const for receive operations during handshake */
        static const int handshakeReceiveTimeout;

        /** The timeout const for send operations during handshake */
        static const int handshakeSendTimeout;

        /**
         * forbidden ctor.
         */
        RenderNetUtil(void);

        /**
         * forbidden dtor.
         */
        ~RenderNetUtil(void);

    };

} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_RENDERNETUTIL_H_INCLUDED */
