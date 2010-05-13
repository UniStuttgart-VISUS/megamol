/*
 * ClusterControllerClient.h
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTERCONTROLLERCLIENT_H_INCLUDED
#define MEGAMOLCORE_CLUSTERCONTROLLERCLIENT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallerSlot.h"
#include "cluster/ClusterController.h"
#include "Module.h"
#include "AbstractSlot.h"
#include "param/ParamSlot.h"
#include "vislib/AbstractCommChannel.h"
#include "vislib/AbstractSimpleMessage.h"
#include "vislib/DiscoveryService.h"
#include "vislib/IPEndPoint.h"
#include "vislib/RunnableThread.h"
#include "vislib/SimpleMessageDispatcher.h"
#include "vislib/SimpleMessageDispatchListener.h"
#include "vislib/SmartRef.h"
#include "vislib/TcpCommChannel.h"


namespace megamol {
namespace core {
namespace cluster {

    /**
     * Interface class for clients of cluster controllers
     */
    class ClusterControllerClient : public AbstractSlot::Listener,
        public vislib::net::SimpleMessageDispatchListener {
    public:

        /** ClusterController will de/register itself */
        friend class ClusterController;

        /** The user message to query the head node in the cluster */
        static const UINT32 USRMSG_QUERYHEAD
            = vislib::net::cluster::DiscoveryService::MSG_TYPE_USER + 0;

        /** The user message to query the head node in the cluster */
        static const UINT32 USRMSG_HEADHERE
            = vislib::net::cluster::DiscoveryService::MSG_TYPE_USER + 1;

        /**
         * Ctor
         */
        ClusterControllerClient(void);

        /**
         * Dtor.
         */
        virtual ~ClusterControllerClient(void);

        /**
         * Informs the client that the cluster is now available.
         */
        virtual void OnClusterAvailable(void);

        /**
         * Informs the client that the cluster is no longer available.
         */
        virtual void OnClusterUnavailable(void);

        /**
         * Informs the client that a new node has been found in the
         * cluster.
         *
         * @param hPeer The peer of the new node
         */
        virtual void OnNodeFound(
            const ClusterController::PeerHandle& hPeer);

        /**
         * Informs the client that a node was lost from the cluster.
         *
         * @param hPeer The peer of the lost node
         */
        virtual void OnNodeLost(
            const ClusterController::PeerHandle& hPeer);

        /**
         * A message has been received.
         *
         * @param hPeer The peer which sent the message
         * @param msgType The type value of the message
         * @param msgBody The data of the message
         */
        virtual void OnUserMsg(const ClusterController::PeerHandle& hPeer,
            const UINT32 msgType, const BYTE *msgBody) = 0;

        /**
         * Sends a message to all nodes in the cluster.
         *
         * @param msgType The type value for the message
         * @param msgBody The data of the message
         * @param msgSize The size of the data of the message in bytes
         */
        void SendUserMsg(const UINT32 msgType, const BYTE *msgBody,
            const SIZE_T msgSize);

        /**
         * Sends a message to one nodes in the cluster.
         *
         * @param hPeer The peer to the node to send the message to
         * @param msgType The type value for the message
         * @param msgBody The data of the message
         * @param msgSize The size of the data of the message in bytes
         */
        void SendUserMsg(const ClusterController::PeerHandle& hPeer,
            const UINT32 msgType, const BYTE *msgBody,
            const SIZE_T msgSize);

    protected:

        /**
         * This method is called when an object connects to the slot.
         *
         * @param slot The slot that triggered this event.
         */
        virtual void OnConnect(AbstractSlot& slot);

        /**
         * This method is called when an object disconnects from the slot.
         *
         * @param slot The slot that triggered this event.
         */
        virtual void OnDisconnect(AbstractSlot& slot);

        /** The caller to register the client at the controller */
        CallerSlot registerSlot;

        /** Slot specifying the communication address */
        param::ParamSlot ctrlCommAddressSlot;

        /**
         * Update callback when the control communication address changes
         *
         * @param address The new address string to be used
         */
        virtual void OnCtrlCommAddressChanged(const vislib::TString& address) = 0;

        /**
         * Stops the control message communication
         */
        void stopCtrlComm(void);

        /**
         * Starts a server end for the control message communication
         *
         * @return A reference to the server communication channel
         */
        vislib::SmartRef<vislib::net::TcpCommChannel> startCtrlCommServer(void);

        /**
         * Starts a client end for the control message communication
         *
         * @param address The address of the server
         *
         * @return The comm channel on success
         */
        vislib::net::AbstractCommChannel * startCtrlCommClient(const vislib::net::IPEndPoint& address);

        /**
         * asnwers if the control message communication channel is connected to the given address.
         *
         * @param address The ip addess including port to test
         *
         * @return True if the control message communication channel is connected to the given address.
         */
        bool isCtrlCommConnectedTo(const char *address) const;

    private:

        /**
         * Update callback when the control communication address changes
         *
         * @param slot Must be 'ctrlCommAddressSlot'
         *
         * @return true
         */
        bool onCtrlCommAddressChanged(param::ParamSlot& slot);

        /** The cluster controller */
        ClusterController *ctrlr;

        /** The control command communication channel */
        vislib::SmartRef<vislib::net::TcpCommChannel> ctrlChannel;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTERCONTROLLERCLIENT_H_INCLUDED */
