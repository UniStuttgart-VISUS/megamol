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

#include "cluster/ClusterController.h"
#include "mmcore/AbstractSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/Listenable.h"
#include "vislib/net/DiscoveryService.h"


namespace megamol::core::cluster {

/**
 * client class for cluster controller
 */
class ClusterControllerClient : public AbstractSlot::Listener, public vislib::Listenable<ClusterControllerClient> {
public:
    /** ClusterController will de/register itself */
    friend class ClusterController;

    /** The user message to query the head node in the cluster */
    static const UINT32 USRMSG_QUERYHEAD = vislib::net::cluster::DiscoveryService::MSG_TYPE_USER + 0;

    /** The user message to query the head node in the cluster */
    static const UINT32 USRMSG_HEADHERE = vislib::net::cluster::DiscoveryService::MSG_TYPE_USER + 1;

    /** The user message to shut down all cluster nodes */
    static const UINT32 USRMSG_SHUTDOWN = vislib::net::cluster::DiscoveryService::MSG_TYPE_USER + 2;

    /**
     * Class for listener object
     */
    class Listener : public vislib::Listenable<ClusterControllerClient>::Listener {
    public:
        /** Ctor */
        Listener() {}

        /** Dtor */
        ~Listener() override {}

        /**
         * Informs the client that the cluster is now available.
         *
         * @param sender The sending object
         */
        virtual void OnClusterDiscoveryAvailable(ClusterControllerClient& sender) {}

        /**
         * Informs the client that the cluster is no longer available.
         *
         * @param sender The sending object
         */
        virtual void OnClusterDiscoveryUnavailable(ClusterControllerClient& sender) {}

        /**
         * Informs the client that a new node has been found in the
         * cluster.
         *
         * @param sender The sending object
         * @param hPeer The peer of the new node
         */
        virtual void OnClusterNodeFound(ClusterControllerClient& sender, const ClusterController::PeerHandle& hPeer) {}

        /**
         * Informs the client that a node was lost from the cluster.
         *
         * @param sender The sending object
         * @param hPeer The peer of the lost node
         */
        virtual void OnClusterNodeLost(ClusterControllerClient& sender, const ClusterController::PeerHandle& hPeer) {}

        /**
         * A message has been received.
         *
         * @param sender The sending object
         * @param hPeer The peer which sent the message
         * @param msgType The type value of the message
         * @param msgBody The data of the message
         */
        virtual void OnClusterUserMessage(ClusterControllerClient& sender, const ClusterController::PeerHandle& hPeer,
            bool isClusterMember, const UINT32 msgType, const BYTE* msgBody) {}
    };

    /**
     * Ctor
     */
    ClusterControllerClient();

    /**
     * Dtor.
     */
    ~ClusterControllerClient() override;

    /**
     * Sends a message to all nodes in the cluster.
     *
     * @param msgType The type value for the message
     * @param msgBody The data of the message
     * @param msgSize The size of the data of the message in bytes
     */
    void SendUserMsg(const UINT32 msgType, const BYTE* msgBody, const SIZE_T msgSize);

    /**
     * Sends a message to one nodes in the cluster.
     *
     * @param hPeer The peer to the node to send the message to
     * @param msgType The type value for the message
     * @param msgBody The data of the message
     * @param msgSize The size of the data of the message in bytes
     */
    void SendUserMsg(
        const ClusterController::PeerHandle& hPeer, const UINT32 msgType, const BYTE* msgBody, const SIZE_T msgSize);

    /**
     * Access the caller to register the client at the controller
     *
     * @return The caller to register the client at the controller
     */
    inline CallerSlot& RegisterSlot() {
        return this->registerSlot;
    }

protected:
    /**
     * This method is called when an object connects to the slot.
     *
     * @param slot The slot that triggered this event.
     */
    void OnConnect(AbstractSlot& slot) override;

    /**
     * This method is called when an object disconnects from the slot.
     *
     * @param slot The slot that triggered this event.
     */
    void OnDisconnect(AbstractSlot& slot) override;

    /**
     * Informs the client that the cluster is now available.
     */
    void onClusterAvailable();

    /**
     * Informs the client that the cluster is no longer available.
     */
    void onClusterUnavailable();

    /**
     * Informs the client that a new node has been found in the
     * cluster.
     *
     * @param hPeer The peer of the new node
     */
    void onNodeFound(const ClusterController::PeerHandle& hPeer);

    /**
     * Informs the client that a node was lost from the cluster.
     *
     * @param hPeer The peer of the lost node
     */
    void onNodeLost(const ClusterController::PeerHandle& hPeer);

    /**
     * A message has been received.
     *
     * @param hPeer The peer which sent the message
     * @param msgType The type value of the message
     * @param msgBody The data of the message
     */
    void onUserMsg(
        const ClusterController::PeerHandle& hPeer, bool isClusterMember, const UINT32 msgType, const BYTE* msgBody);

    /** The caller to register the client at the controller */
    CallerSlot registerSlot;

private:
    /** The cluster controller */
    ClusterController* ctrlr;
};


} // namespace megamol::core::cluster

#endif /* MEGAMOLCORE_CLUSTERCONTROLLERCLIENT_H_INCLUDED */
