/*
 * ClusterDiscoveryService.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CLUSTERDISCOVERYSERVICE_H_INCLUDED
#define VISLIB_CLUSTERDISCOVERYSERVICE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/IPAddress.h"   // Must be included at begin!
#include "vislib/Socket.h"      // Must be included at begin!
#include "vislib/Array.h"
#include "vislib/CriticalSection.h"
#include "vislib/String.h"
#include "vislib/Thread.h"
#include "vislib/types.h"



namespace vislib {
namespace net {


    class ClusterDiscoveryService {

    public:

        /**
         * The maximum size of user data and the cluster name that can be 
         * used. 
         */
        static const SIZE_T MAX_USER_DATA = 256;

        ClusterDiscoveryService(const StringA& name, 
            const SocketAddress& bindAddr, const IPAddress& bcastAddr);

        virtual ~ClusterDiscoveryService(void);

        virtual bool Start(void);

    protected:

        typedef struct Message_t {
            UINT16 magicNumber;
            UINT16 msgType;
            union {
                struct sockaddr_in inetAddr;
                char name[MAX_USER_DATA];
                BYTE userData[MAX_USER_DATA];
            };
        } Message;

        /** 
         * This structure is used to identify a peer node that has been found
         * during the discovery process.
         */
        typedef struct PeerNode_t {
            IPAddress ipAddress;

            inline bool operator ==(const PeerNode_t& rhs) const {
                return (this->ipAddress == rhs.ipAddress);
            }
        } PeerNode;

        /**
         * This is the worker function of the thread that periodically sends 
         * discovery requests to other nodes.
         *
         * @param userData A pointer to the ClusterDiscoveryService running the
         *                 thread.
         *
         * @return 0 in case of success,
         */
        static DWORD requestFunc(const void *userData);

        /**
         * This is the worker function of the thread that receives discovery
         * requests from other computers and answers those.
         *
         * @param userData A pointer to the ClusterDiscoveryService running the
         *                 thread.
         *
         * @return 0 in case of success,
         */
        static DWORD responseFunc(const void *userData);

        /** The magic number at the begin of each message. */
        static const UINT16 MAGIC_NUMBER;

        /** Message type ID of a discovery request. */
        static const UINT16 MSG_TYPE_DISCOVERY_REQUEST;

        /** Message type ID of a discovery response. */
        static const UINT16 MSG_TYPE_DISCVOERY_RESPONSE;

        /** This is the broadcast address to send requests to. */
        SocketAddress bcastAddr;

        /** The name of the cluster this discovery service should form. */
        StringA name;

        /** The thread performing the node discovery. */
        sys::Thread requestThread;

        /** The thread answering discovery requests. */
        sys::Thread responseThread;

        /** This array holds the peer nodes. */
        Array<PeerNode> peerNodes;

        /** Critical section protecting access to the 'peerNodes' array. */
        sys::CriticalSection peerNodesCritSect;
    };


} /* end namespace net */
} /* end namespace vislib */

#endif /* VISLIB_CLUSTERDISCOVERYSERVICE_H_INCLUDED */
