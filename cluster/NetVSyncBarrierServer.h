/*
 * NetVSyncBarrierServer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_NETVSYNCBARRIERSERVER_H_INCLUDED
#define MEGAMOLCORE_NETVSYNCBARRIERSERVER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "AbstractNamedObject.h"
#include "cluster/CommChannelServer.h"
#include "vislib/CriticalSection.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/String.h"
#include "vislib/IPEndPoint.h"


namespace megamol {
namespace core {
namespace cluster {


    /**
     * The network v-sync barrier server
     */
    class NetVSyncBarrierServer : protected CommChannelServer::Listener {
    public:

        /**
         * Ctor.
         *
         * @param owner The owning object
         */
        NetVSyncBarrierServer(AbstractNamedObject *owner);

        /**
         * ~Dtor.
         */
        virtual ~NetVSyncBarrierServer(void);

        /**
         * Starts the network barrier server on the specified local end point
         *
         * @param lep The local end point address string
         *
         * @return True on success
         *
         * @throws vislib::Exception on any critical error
         */
        bool Start(vislib::StringA lep);

        /**
         * Stopps the network barrier server
         */
        void Stop(void) throw();

        /**
         * Sets the name of the view
         *
         * @param viewname The name of the view
         */
        inline void SetViewName(const vislib::StringA& viewname) {
            this->viewName = viewname;
        }

    protected:

        /**
         * Informs that the control channel is now connected an can send and receive messages
         *
         * @param sender The sending object
         */
        virtual void OnCommChannelServerStarted(CommChannelServer& server);

        /**
         * Informs that the control channel is no longer connected.
         *
         * @param sender The sending object
         */
        virtual void OnCommChannelServerStopped(CommChannelServer& server);

        /**
         * Informs that the control channel is now connected an can send and receive messages
         *
         * @param sender The sending object
         * @param channel The communication channel
         */
        virtual void OnCommChannelConnect(CommChannelServer& server, CommChannel& channel);

        /**
         * Informs that the control channel is no longer connected.
         *
         * @param sender The sending object
         * @param channel The communication channel
         */
        virtual void OnCommChannelDisconnect(CommChannelServer& server, CommChannel& channel);

        /**
         * A message has been received over the control channel.
         *
         * @param sender The sending object
         * @param channel The communication channel
         * @param msg The received message
         */
        virtual void OnCommChannelMessage(CommChannelServer& server, CommChannel& channel,
                const vislib::net::AbstractSimpleMessage& msg);

    private:

        /**
         * Checks if the barrier can be crossed
         */
        inline void checkBarrier(void);

        /** The server */
        cluster::CommChannelServer server;

        /** The local server end point */
        vislib::net::IPEndPoint serverEndpoint;

        /** The peers */
        vislib::SingleLinkedList<void *> peers;

        /** The current barrier */
        unsigned char currentBarrier;

        /** The number of peers waiting at the barrier */
        SIZE_T waitingPeerCount;

        /** The synchronization lock object */
        vislib::sys::CriticalSection lock;

        /** The core instance */
        AbstractNamedObject *owner;

        /** The name of the view */
        vislib::StringA viewName;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_NETVSYNCBARRIERSERVER_H_INCLUDED */
