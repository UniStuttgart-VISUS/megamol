/*
 * ClusterController.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTERCONTROLLER_H_INCLUDED
#define MEGAMOLCORE_CLUSTERCONTROLLER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "job/AbstractJobThread.h"
#include "CalleeSlot.h"
#include "Module.h"
#include "param/ParamSlot.h"
#include "vislib/CriticalSection.h"
#include "vislib/DiscoveryListener.h"
#include "vislib/DiscoveryService.h"
#include "vislib/forceinline.h"
#include "vislib/IPAddress.h"
#include "vislib/IPAddress6.h"


namespace megamol {
namespace core {
namespace special {

    /** forward declaration */
    class ClusterControllerClient;


    /**
     * Class implementing the cluster rendering master job
     */
    class ClusterController : public job::AbstractJobThread, public Module,
        public vislib::net::cluster::DiscoveryListener {
    public:

        /** DiscoveryService::PeerHandle */
        typedef vislib::net::cluster::DiscoveryService::PeerHandle PeerHandle;

        /** The default name of the rendering cluster */
        static const char * DEFAULT_CLUSTERNAME;

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ClusterController";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
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
        virtual ~ClusterController();

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

        /**
         * Perform the work of a thread.
         *
         * @param userData A pointer to user data that are passed to the thread,
         *                 if it started.
         *
         * @return The application dependent return code of the thread. This 
         *         must not be STILL_ACTIVE (259).
         */
        virtual DWORD Run(void *userData);

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
        virtual void OnNodeFound(
            const vislib::net::cluster::DiscoveryService& src,
            const vislib::net::cluster::DiscoveryService::PeerHandle& hPeer)
            throw();

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
        virtual void OnNodeLost(
            const vislib::net::cluster::DiscoveryService& src,
            const vislib::net::cluster::DiscoveryService::PeerHandle& hPeer,
            const vislib::net::cluster::DiscoveryListener::NodeLostReason
                reason) throw();

        /**
         * This method is
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the discovery thread.
         *
         * @param src     The discovery service that fired the event.
         * @param hPeer   Handle of the peer node that sent the message.
         * @param msgType The message type identifier.
         * @param msgBody The message body data. These are user defined and
         *                probably dependent on the 'msgType'. The caller 
         *                remains owner of the memory designated by 'msgBody'.
         */
        virtual void OnUserMessage(
            const vislib::net::cluster::DiscoveryService& src,
            const vislib::net::cluster::DiscoveryService::PeerHandle& hPeer, 
            const UINT32 msgType, const BYTE *msgBody) throw();

    private:

        /**
         * Stops the discovery service.
         */
        void stopDiscoveryService(void);

        ///**
        // * Gets the network configuration parameters for the cluster discovery
        // * service
        // *
        // * @param addr4    May receive the IPv4 adapter address
        // * @param bcast4   May receive the IPv4 broadcast address
        // * @param ip4Valid Indicates if 'addr4' and 'bcast4' are valid
        // * @param addr6    May receive the IPv6 adapter address
        // * @param bcast6   May receive the IPv6 broadcast address
        // * @param ip6Valid Indicates if 'addr6' and 'bcast6' are valid
        // * @param port     The udp socket port to be used.
        // *
        // * @return (ip4Valid || ip6Valid) for convenience
        // */
        //bool getServiceNetConfig(vislib::net::IPAddress& addr4,
        //    vislib::net::IPAddress& bcast4, bool& ip4Valid,
        //    vislib::net::IPAddress6& addr6, vislib::net::IPAddress6& bcast6,
        //    bool& ip6Valid, unsigned short& port);

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

        /** The name of the rendering cluster */
        param::ParamSlot clusterName;

        /**
         * The address of the adapter to be used by the cluster discovery
         * service
         */
        param::ParamSlot cdsAdapterAddress;

        /**
         * The broadcast address to be used by the cluster discovery service.
         */
        param::ParamSlot cdsBCastAddress;

        /** The port to be used by the cluster discovery service. */
        param::ParamSlot cdsPort;

        /** Flag to start or stop the cluster discovery service */
        param::ParamSlot cdsRun;

        /** The discovery service object */
        vislib::net::cluster::DiscoveryService discoveryService;

        /** The slot to register at */
        CalleeSlot registerSlot;

        /** List of the registered clients */
        vislib::SingleLinkedList<ClusterControllerClient *> clients;

        /** The locking for accessing the clients list */
        vislib::sys::CriticalSection clientsLock;

    };


} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTERCONTROLLER_H_INCLUDED */
