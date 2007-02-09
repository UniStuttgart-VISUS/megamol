/*
 * ClusterDiscoveryListener.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CLUSTERDISCOVERYLISTENER_H_INCLUDED
#define VISLIB_CLUSTERDISCOVERYLISTENER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/types.h"


namespace vislib {
namespace net {

    /* Forward declarations. */
    class ClusterDiscoveryService;
    class SocketAddress;


    /**
     * Subclasses of this class can be informed by a ClusterDiscoveryService
     * about new nodes that belong to the same cluster. 
     */
    class ClusterDiscoveryListener {

    public:

		/**
		 * Possible reasons for an OnNodeLost notification:
		 *
		 * LOST_EXPLICITLY means that the peer node explicitly disconnected by
		 * sending the sayonara message.
		 *
		 * LOST_IMLICITLY means that the peer node was removed because it did not
		 * properly answer a alive request.
		 */
		enum NodeLostReason{
			LOST_EXPLICITLY = 1,
			LOST_IMLICITLY
		};

        /** Ctor. */
        ClusterDiscoveryListener(void);

        /** Dtor. */
        virtual ~ClusterDiscoveryListener(void);

        /**
         * This method will be called, if a new computer was found
         * by a ClusterDiscoveryService.
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the discovery thread.
         *
         * @param src  The discovery service that fired the event.
         * @param addr The response address that the node discovered has
         *             specified.
         */
        virtual void OnNodeFound(const ClusterDiscoveryService& src, 
            const SocketAddress& addr) = 0;

        /**
         * This method will be called, if a new computer disconnected from
         * a ClusterDiscoveryService or is regarded as lost.
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the discovery thread.
         *
         * @param src    The discovery service that fired the event.
         * @param addr   The response address that the node that left the 
         *               cluster had.
         * @param reason The reason why the node was removed from the cluster.
         */
		virtual void OnNodeLost(const ClusterDiscoveryService& src,
			const SocketAddress& addr, const NodeLostReason reason) = 0;

        /**
         * This method is
         *
         * This method should return very quickly and should not perform
         * excessive work as it is executed in the discovery thread.
         *
         * @param src     The discovery service that fired the event.
         * @param sender  The address of the node that sent the message.
         * @param msgType The message type identifier.
         * @param msgBody The message body data. These are user defined and
         *                probably dependent on the 'msgType'. The caller 
         *                remains owner of the memory designated by 'msgBody'.
         */
        virtual void OnUserMessage(const ClusterDiscoveryService& src,
            const SocketAddress& sender, const UINT16 msgType, 
            const BYTE *msgBody);
    };
    
} /* end namespace net */
} /* end namespace vislib */

#endif /* VISLIB_CLUSTERDISCOVERYLISTENER_H_INCLUDED */
