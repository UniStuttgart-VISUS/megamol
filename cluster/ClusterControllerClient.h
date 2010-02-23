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


namespace megamol {
namespace core {
namespace cluster {

    /**
     * Interface class for clients of cluster controllers
     */
    class ClusterControllerClient {
    public:

        /** ClusterController will de/register itself */
        friend class ClusterController;

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
            const UINT32 msgType, const BYTE *msgBody);

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

        /** The caller to register the client at the controller */
        CallerSlot registerSlot;

    private:

        /** The cluster controller */
        ClusterController *ctrlr;

    };


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTERCONTROLLERCLIENT_H_INCLUDED */
