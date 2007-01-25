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

        /** The first message ID that can be used for a user message. */
        static const UINT16 MSG_TYPE_USER;

        ClusterDiscoveryService(const StringA& name, 
            const SocketAddress& bindAddr, const IPAddress& bcastAddr,
            const SocketAddress& responseAddr);

        /** 
         * Dtor.
         *
         * Note that the dtor will terminate the discovery.
         */
        virtual ~ClusterDiscoveryService(void);


        /**
         * Answer the address the service is listening on for discovery
         * requests.
         *
         * @return The address the listening socket is bound to.
         */
        const SocketAddress& GetBindAddr(void) const {
            return this->bindAddr;
        }

        /**
         * Answer the cluster identifier that is used for discovery.
         *
         * @return The name.
         */
        inline const StringA& GetName(void) const {
            return this->name;
        }

        /**
         * Answer the call back socket address that is sent to peer nodes 
         * when they are discovered. This address can be used to establish a
         * connection to our node in a application defined manner.
         *
         * @return The address sent as response.
         */
        const SocketAddress& GetResponseAddr(void) const {
            return this->responseAddr;
        }

        virtual bool Start(void);

        virtual bool Stop(void);

    protected:

        /**
         * This Runnable is the worker that broadcasts the discovery requests of
         * for a specific ClusterDiscoveryService.
         */
        class Requester : public vislib::sys::Runnable {

        public:

            /**
             * Create a new instance that is working for 'cds'.
             *
             * @param cds The ClusterDiscoveryService that determines the 
             *            communication parameters and receives the peer nodes
             *            that have been detected.
             */
            Requester(ClusterDiscoveryService& cds);

            /** Dtor. */
            virtual ~Requester(void);

            /**
             * Performs the discovery.
             *
             * @param reserved Reserved. Must be NULL.
             *
             * @return 0, if the work was successfully finished, an error code
             *         otherwise.
             */
            virtual DWORD Run(const void *reserved = NULL);

            /**
             * Ask the thread to terminate.
             *
             * @return true.
             */
            virtual bool Terminate(void);

        private:

            /** The discovery service the thread is working for. */
            ClusterDiscoveryService& cds;

            /** Flag for terminating the thread safely. */
            bool isRunning;
        }; /* end class Requester */


        /**
         * This Runnable receives discovery requests from other nodes and 
         * answers these requests on behalf of a ClusterDiscoveryService.
         */
        class Responder : public vislib::sys::Runnable {

        public:

            /**
             * Create a new instance answering discovery requests directed to
             * 'cds'.
             *
             * @param cds The ClusterDiscoveryService to work for.
             */
            Responder(ClusterDiscoveryService& cds);

            /** Dtor. */
            virtual ~Responder(void);

            /**
             * Answers the discovery requests.
             *
             * @param reserved Reserved. Must be NULL.
             *
             * @return 0, if the work was successfully finished, an error code
             *         otherwise.
             */
            virtual DWORD Run(const void *reserved = NULL);

            /**
             * Ask the thread to terminate.
             *
             * @return true.
             */
            virtual bool Terminate(void);

        private:

            /** The discovery service the thread is working for. */
            ClusterDiscoveryService& cds;

            /** Flag for terminating the thread safely. */
            bool isRunning;
        }; /* end class Responder */

        /**
         * This structure is sent as message by the discovery service. Only one 
         * type of message is used as we cannot know the order and size of
         * UDP datagrams in advance.
         */
        typedef struct Message_t {
            UINT16 magicNumber;                     // Must be MAGIC_NUMBER.
            UINT16 msgType;                         // The type identifier.
            union {
                struct sockaddr_in responseAddr;    // Peer address to store.
                char name[MAX_USER_DATA];           // Name of searched cluster.
                BYTE userData[MAX_USER_DATA];       // User defined data.
            };
        } Message;

        /** 
         * This structure is used to identify a peer node that has been found
         * during the discovery process.
         */
        typedef struct PeerNode_t {
            SocketAddress address;

            inline bool operator ==(const PeerNode_t& rhs) const {
                return (this->address == rhs.address);
            }
        } PeerNode;

        /** The magic number at the begin of each message. */
        static const UINT16 MAGIC_NUMBER;

        /** Message type ID of a discovery request. */
        static const UINT16 MSG_TYPE_DISCOVERY_REQUEST;

        /** Message type ID of a discovery response. */
        static const UINT16 MSG_TYPE_DISCOVERY_RESPONSE;

        /** This is the broadcast address to send requests to. */
        SocketAddress bcastAddr;

        /** The address that the response thread binds to. */
        SocketAddress bindAddr;

        /** The address we send in a response message. */
        SocketAddress responseAddr;

        /**
         * The number of expected responses. The thread will wait for this
         * number of nodes to anwer before sending another discovery
         * request. 
         */
        UINT expectedResponseCnt;

        /** The time in milliseconds between two discovery requests. */
        UINT requestInterval;

        /** The name of the cluster this discovery service should form. */
        StringA name;

        /** The worker object of 'requestThread'. */
        Requester *requester;

        /** The worder object of 'responseThread'. */
        Responder *responder;

        /** The thread performing the node discovery. */
        sys::Thread *requestThread;

        /** The thread answering discovery requests. */
        sys::Thread *responseThread;

        /** This array holds the peer nodes. */
        Array<PeerNode> peerNodes;

        /** Critical section protecting access to the 'peerNodes' array. */
        sys::CriticalSection peerNodesCritSect;

        /** The timeout for receive operations in milliseconds. */
        INT timeoutReceive;

        /** The timeout for send operations in milliseconds. */
        INT timeoutSend;
    };


} /* end namespace net */
} /* end namespace vislib */

#endif /* VISLIB_CLUSTERDISCOVERYSERVICE_H_INCLUDED */
