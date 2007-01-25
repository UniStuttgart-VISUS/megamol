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

        /** Ctor. */
        ClusterDiscoveryListener(void);

        /** Dtor. */
        virtual ~ClusterDiscoveryListener(void);

        /**
         * This method will be called, if a new computer was found
         * by a ClusterDiscoveryService.
         *
         * @param src  The discovery service that fired the event.
         * @param addr The response address that the node discovered has
         *             specified.
         */
        virtual void OnNodeFound(const ClusterDiscoveryService& src, 
            const SocketAddress& addr) = 0;

    };
    
} /* end namespace net */
} /* end namespace vislib */

#endif /* VISLIB_CLUSTERDISCOVERYLISTENER_H_INCLUDED */
