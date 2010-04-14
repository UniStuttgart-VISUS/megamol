/*
 * DiscoveryListener.h
 *
 * Copyright (C) 2006 - 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_DISCOVERYLISTENER_H_INCLUDED
#define VISLIB_DISCOVERYLISTENER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/DiscoveryService.h"


namespace vislib {
namespace net {
namespace cluster {


    /**
     * Subclasses of this class can be informed by a DiscoveryService
     * about new nodes that belong to the same cluster. 
     */
    class DiscoveryListener {

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
        DiscoveryListener(void);

        /** Dtor. */
        virtual ~DiscoveryListener(void);

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
        virtual void OnNodeFound(DiscoveryService& src, 
            const DiscoveryService::PeerHandle& hPeer) throw() = 0;

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
        virtual void OnNodeLost(DiscoveryService& src,
            const DiscoveryService::PeerHandle& hPeer, 
            const NodeLostReason reason) throw() = 0;

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
        virtual void OnUserMessage(DiscoveryService& src,
            const DiscoveryService::PeerHandle& hPeer, 
            const bool isClusterMember,
            const UINT32 msgType, const BYTE *msgBody) throw();
    };

} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_DISCOVERYLISTENER_H_INCLUDED */
