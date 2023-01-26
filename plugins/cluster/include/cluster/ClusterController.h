/*
 * ClusterController.h
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTERCONTROLLER_H_INCLUDED
#define MEGAMOLCORE_CLUSTERCONTROLLER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"
#include "mmcore/job/AbstractThreadedJob.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/net/DiscoveryListener.h"
#include "vislib/net/DiscoveryService.h"
#include "vislib/sys/CriticalSection.h"


namespace megamol {
namespace core {
namespace cluster {

/** forward declaration */
class ClusterControllerClient;


/**
 * Class implementing the cluster rendering master job
 */
class ClusterController : public job::AbstractThreadedJob,
                          public Module,
                          public vislib::net::cluster::DiscoveryListener {
public:
    /** DiscoveryService::PeerHandle */
    typedef vislib::net::cluster::DiscoveryService::PeerHandle PeerHandle;

    /** The default name of the rendering cluster */
    static const char* DEFAULT_CLUSTERNAME;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "ClusterController";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "The controller thread for cluster operation";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /**
     * Ctor
     */
    ClusterController();

    /**
     * Dtor
     */
    ~ClusterController() override;

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
    void SendUserMsg(const PeerHandle& hPeer, const UINT32 msgType, const BYTE* msgBody, const SIZE_T msgSize);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create(void) override;

    /**
     * Implementation of 'Release'.
     */
    void release(void) override;

    /**
     * Perform the work of a thread.
     *
     * @param userData A pointer to user data that are passed to the thread,
     *                 if it started.
     *
     * @return The application dependent return code of the thread. This
     *         must not be STILL_ACTIVE (259).
     */
    DWORD Run(void* userData) override;

    /**
     * This method will be called, if a new computer was found
     * by a DiscoveryService.
     *
     * This method should return very quickly and should not perform
     * excessive work as it is executed in the discovery thread.
     *
     * @param src   The discovery service that fired the event.
     * @param hPeer The handle of the peer that was found. The response
     *              address associated with this handle can be retrieved
     *              via src[hPeer].
     */
    void OnNodeFound(vislib::net::cluster::DiscoveryService& src,
        const vislib::net::cluster::DiscoveryService::PeerHandle& hPeer) throw() override;

    /**
     * This method will be called, if a new computer disconnected from
     * a DiscoveryService or is regarded as lost.
     *
     * This method should return very quickly and should not perform
     * excessive work as it is executed in the discovery thread.
     *
     * Note that the peer handle 'hPeer' is only guaranteed to be valid
     * until this method returns!
     *
     * @param src    The discovery service that fired the event.
     * @param hPeer  The handle of the peer that was removed.
     * @param reason The reason why the node was removed from the cluster.
     */
    void OnNodeLost(vislib::net::cluster::DiscoveryService& src,
        const vislib::net::cluster::DiscoveryService::PeerHandle& hPeer,
        const vislib::net::cluster::DiscoveryListener::NodeLostReason reason) throw() override;
    /**
     * This method is called once the discovery service receives a user
     * message (user defined payload).
     *
     * This method should return very quickly and should not perform
     * excessive work as it is executed in the discovery thread.
     *
     * Remarks regarding 'hPeer': The peer handle can be used to send an
     * answer message to the sender of this message. It can be used for
     * other operations on 'src' in case the peer node is a member of the
     * cluster, i. e. if 'isClusterMember' is true. Otherwise, operations
     * on 'src' might fail and indicate an invalid handle. It is guaranteed
     * that SendUserMessage() will work for non-member handles.
     *
     * @param src             The discovery service that fired the event.
     * @param hPeer           Handle of the peer node that sent the message.
     * @param isClusterMember This flag is set if the node designated by
     *                        'hPeer' is a known peer node of the cluster
     *                        managed by 'src'. If not, the message was
     *                        received from an observer node. Please be
     *                        aware that 'hPeer' is of limited use in this
     *                        case.
     * @param msgType         The message type identifier.
     * @param msgBody         The message body data. These are user defined
     *                        and probably dependent on the 'msgType'. The
     *                        callee remains owner of the memory designated
     *                        by 'msgBody'. It is valid until the callback
     *                        is left.
     */
    void OnUserMessage(vislib::net::cluster::DiscoveryService& src,
        const vislib::net::cluster::DiscoveryService::PeerHandle& hPeer, const bool isClusterMember,
        const UINT32 msgType, const BYTE* msgBody) throw() override;

private:
    /**
     * Answer the default port
     *
     * @return The default port
     */
    UINT16 defaultPort(void);

    /**
     * Stops the discovery service.
     */
    void stopDiscoveryService(void);

    /**
     * A module want's to register at the controller.
     *
     * @param call The calling call.
     *
     * @return 'true'
     */
    bool registerModule(Call& call);

    /**
     * A module want's to unregister from the controller.
     *
     * @param call The calling call.
     *
     * @return 'true'
     */
    bool unregisterModule(Call& call);

    /**
     * A module queries the status of the controller
     *
     * @param call The calling call.
     *
     * @return 'true'
     */
    bool queryStatus(Call& call);

    /**
     * Event handler for the shutdown-cluster button
     *
     * @param slot Must be shutdownClusterButton
     *
     * @return true
     */
    bool onShutdownCluster(param::ParamSlot& slot);

    /** The name of the rendering cluster */
    param::ParamSlot cdsNameSlot;

    /** The ip port to be used by the cluster discovery service. */
    param::ParamSlot cdsPortSlot;

    /** Flag to start or stop the cluster discovery service */
    param::ParamSlot cdsRunSlot;

    /** Button slot to shut down the whole cluster */
    param::ParamSlot shutdownClusterSlot;

    /** The discovery service object */
    vislib::net::cluster::DiscoveryService discoveryService;

    /** The slot to register at */
    CalleeSlot registerSlot;

    /** List of the registered clients */
    vislib::SingleLinkedList<ClusterControllerClient*> clients;

    /** The locking for accessing the clients list */
    vislib::sys::CriticalSection clientsLock;
};


} // namespace cluster
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTERCONTROLLER_H_INCLUDED */
