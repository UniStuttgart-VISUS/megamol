/*
 * DiscoveryService.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CLUSTERDISCOVERYSERVICE_H_INCLUDED
#define VISLIB_CLUSTERDISCOVERYSERVICE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/IPAgnosticAddress.h"   // Must be included at begin!
#include "vislib/Array.h"
#include "vislib/CriticalSection.h"
#include "vislib/Interlocked.h"
#include "vislib/RunnableThread.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/Socket.h"
#include "vislib/SmartPtr.h"
#include "vislib/String.h"
#include "vislib/types.h"



namespace vislib {
namespace net {
namespace cluster {

    /* Forward declarations. */
    class DiscoveryListener;


    /**
     * This class implements a method for discovering other computers in a
     * network via UDP broadcasts. 
     *
     * The user specifies a name as identifier of the cluster to be searched
     * and the object creates an array of all nodes that respond to a request
     * whether they are also members of this cluster. The object also anwers
     * requests of other nodes.
     *
     * The vislib::net::cluster::DiscoveryService supersedes 
     * vislib::net::ClusterDiscoveryService. However, the protocol is 
     * compatible.
     */
    class DiscoveryService {

    private:

        /* Forward declarations. */
        class DiscoveryConfigEx;


        /**
         * This is the opaque peer node handle that contains the information
         * that the discovery service needs for each known peer node. 
         * 
         * A smart pointer to these classes is exposed as handle to the user of
         * the discovery service. However, no assumption should be made about
         * the internal structure of the peer node handle.
         */
        class PeerNode {

        public:

            /**
             * Default ctor.
             *
             * This is required for storing peer nodes in a VISlib array.
             */
            PeerNode(void);

            /**
             * Create new peer node.
             *
             * @param discoveryAddress
             * @param responseAddress
             * @param cntResponseChances
             * @param discoverySource
             */
            PeerNode(const IPEndPoint& discoveryAddress, 
                const IPEndPoint& responseAddress,
                const UINT cntResponseChances, 
                DiscoveryConfigEx *discoverySource);

            /**
             * Clone 'rhs'.
             *
             * @param rhs
             */
            inline PeerNode(const PeerNode& rhs) {
                *this = rhs;
            }

            PeerNode& operator =(const PeerNode& rhs);

            inline bool operator ==(const PeerNode& rhs) const {
                // Note: The application logic of the peer list defines that the
                // 'responseAddress' member is primary key of this object.
                return (this->responseAddress == rhs.responseAddress);
            }

            inline bool operator !=(const PeerNode& rhs) const {
                return !(*this == rhs);
            }

        private:

            /**
             * Decrement the response chance counter and detect whether the 
             * counter is still positive. It is safe to call the method if the
             * counter is already zero. The call has no effect in this case.
             *
             * @return true if the counter did not yet reach zero, 
             *         false if it is zero (after the decrement).
             */
            bool decrementResponseChances(void);

            /**
             * Answer the discovery address of the peer node (if IPv4).
             *
             * @return The IPv4 address of the peer node.
             *
             * @throws IllegalStateException If the peer node is not IPv4.
             */
            inline IPAddress getDiscoveryAddress4(void) const {
                return this->discoveryAddress.GetIPAddress4();
            }

            /**
             * Answer the discovery address of the peer node.
             *
             * @return The IPv6 address of the peer node.
             */
            inline IPAddress6 getDiscoveryAddress6(void) const {
                return this->discoveryAddress.GetIPAddress6();
            }

            inline const DiscoveryConfigEx& getDiscoverySource(void) const {
                ASSERT(this->discoverySource != NULL);
                return *this->discoverySource;
            }

            inline IPAddress getResponseAddress4(void) const {
                return this->responseAddress.GetIPAddress4();
            }

            inline IPAddress6 getResponseAddress6(void) const {
                return this->responseAddress.GetIPAddress6();
            }

            /**
             * Invalidate the peer node by destroying the user communication 
             * address (ID).
             */
            void invalidate(void);

            /**
             * Check whether the peer node is valid.
             *
             * @return true if the peer node is valid, false otherwise.
             */
            bool isValid(void) const;

            /** Implicit disconnect detection counter. */
            UINT cntResponseChances;

            /** Discovery service address of the peer node (remote address). */
            IPEndPoint discoveryAddress;

            /** The source that discovered this peer node. */
            DiscoveryConfigEx *discoverySource;

            /** User communication address (ID) of the peer node. */
            IPEndPoint responseAddress;

            /** Allow access to all members to enclosing class. */
            friend class DiscoveryService;

        }; /* end class PeerNode */

    public:

        /**
         * A per-adapter discovery configuration.
         *
         * An instance of this class configures the discovery service thread for 
         * a single adapter. There can also be multiple instances on the same 
         * adapter as long as these use different ports or socket sharing is
         * enabled in the discovery service configuration.
         */
        class DiscoveryConfig {

        public:

            DiscoveryConfig(void);

            /**
             * Create a new configuration with all parameters manually 
             * configured.
             *
             * This configuration puts the discovery thread into IPv4 mode.
             *
             * @param responseAddress This is the "call back address" of the 
             *                        current node/adapter, on which 
             *                        user-defined communication should be 
             *                        initiated. The DiscoveryService does not 
             *                        use this address itself (it is a pure 
             *                        payload for it), but communicates it to
             *                        all other nodes, which then can use it.
             *                        These addresses should uniquely identify
             *                        each process in the cluster, i. e. no node
             *                        should specify the same 'responseAddr' as
             *                        some other does.
             * @param bcastAddress    The broadcast address of the network. All
             *                        requests (alive-messages) will be sent to 
             *                        this address. The destination port of 
             *                        messages is derived 'bindPort'. 
             *                        You can use the 
             *                        vislib::net::NetworkInformation class to
             *                        obtain the broadcast address of your 
             *                        subnet.
             * @param bindPort        The port to bind the receiver thread to.
             *                        All discovery requests are directed to
             *                        this port.
             */
            DiscoveryConfig(const IPEndPoint& responseAddress, 
                const IPAddress& bcastAddress,
                const USHORT bindPort = DEFAULT_PORT);

            /**
             * Create a new configuration with all parameters manually 
             * configured.
             *
             * This configuration puts the discovery thread into IPv6 mode.
             *
             * @param responseAddress This is the "call back address" of the 
             *                        current node/adapter, on which 
             *                        user-defined communication should be 
             *                        initiated. The DiscoveryService does not 
             *                        use this address itself (it is a pure 
             *                        payload for it), but communicates it to
             *                        all other nodes, which then can use it.
             *                        These addresses should uniquely identify
             *                        each process in the cluster, i. e. no node
             *                        should specify the same 'responseAddr' as
             *                        some other does.
             * @param bcastAddress    The broadcast address of the network. All
             *                        requests (alive-messages) will be sent to 
             *                        this address. The destination port of 
             *                        messages is derived 'bindPort'. 
             *                        You can use the 
             *                        vislib::net::NetworkInformation class to
             *                        obtain the broadcast address of your 
             *                        subnet.
             * @param bindPort        The port to bind the receiver thread to.
             *                        All discovery requests are directed to
             *                        this port.
             */
            DiscoveryConfig(const IPEndPoint& responseAddress, 
                const IPAddress6& bcastAddress,
                const USHORT bindPort = DEFAULT_PORT);

            /**
             * Create a new configuration with all parameters manually 
             * configured.
             *
             * The mode of the discovery service depends on the address family
             * of and 'bcastAddress'.
             *
             * @param responseAddress This is the "call back address" of the 
             *                        current node/adapter, on which 
             *                        user-defined communication should be 
             *                        initiated. The DiscoveryService does not 
             *                        use this address itself (it is a pure 
             *                        payload for it), but communicates it to
             *                        all other nodes, which then can use it.
             *                        These addresses should uniquely identify
             *                        each process in the cluster, i. e. no node
             *                        should specify the same 'responseAddr' as
             *                        some other does.
             * @param bcastAddress    The broadcast address of the network. All
             *                        requests (alive-messages) will be sent to 
             *                        this address. The destination port of 
             *                        messages is derived 'bindPort'. 
             *                        You can use the 
             *                        vislib::net::NetworkInformation class to
             *                        obtain the broadcast address of your 
             *                        subnet.
             * @param bindPort        The port to bind the receiver thread to.
             *                        All discovery requests are directed to
             *                        this port.
             */
            DiscoveryConfig(const IPEndPoint& responseAddress, 
                const IPAgnosticAddress& bcastAddress,
                const USHORT bindPort = DEFAULT_PORT);

            /**
             * Create a new configuration with all parameters manually 
             * configured.
             *
             * @param responseAddress
             * @param bindAddress
             * @param bindPort
             */
            //DiscoveryConfig(const IPEndPoint& responseAddress, 
            //    const IPAddress6& bindAddress, 
            //    const USHORT bindPort = DEFAULT_PORT);

            /**
             * Create a new configuration that uses the broadcast address of the
             * subnet 'responseAddress' is in.
             *
             * @param responseAddress This is the "call back address" of the 
             *                        current node/adapter, on which 
             *                        user-defined communication should be 
             *                        initiated. The DiscoveryService does not 
             *                        use this address itself (it is a pure 
             *                        payload for it), but communicates it to
             *                        all other nodes, which then can use it.
             *                        These addresses should uniquely identify
             *                        each process in the cluster, i. e. no node
             *                        should specify the same 'responseAddr' as
             *                        some other does.
             * @param bindPort        The port to bind the receiver thread to.
             *                        All discovery requests are directed to
             *                        this port.
             *
             * @throws IllegalParamException If the broadcast address to send
             *                               alive beacons to cannot be 
             *                               determined from 'responseAddress'.
             */
            DiscoveryConfig(const IPEndPoint& responseAddress,
                const USHORT bindPort = DEFAULT_PORT);

            /**
             * Clone 'rhs'.
             *
             * @param rhs The object to be cloned.
             */
            DiscoveryConfig(const DiscoveryConfig& rhs);

            /**
             * Dtor.
             */
            virtual ~DiscoveryConfig(void);

            /**
             * Answer the broadcast address that alive-messages will be sent to.
             *
             * @return The broadcast address.
             */
            inline const IPEndPoint& GetBcastAddress(void) const {
                return this->bcastAddress;
            }

            /**
             * Answer the end point that the local receiver was requested to 
             * bind to.
             *
             * @return The local socket end point.
             */
            inline const IPEndPoint& GetBindAddress(void) const {
                return this->bindAddress;
            }

            /**
             * Get the address family of the protocol that the discovery thread 
             * is working on.
             *
             * @return The address family that is used for the local socket end 
             *         point.
             */
            inline IPEndPoint::AddressFamily GetAddressFamily(void) const {
                return this->bindAddress.GetAddressFamily();
            }

            /**
             * Get the protocol family that the discovery thread is working on.
             *
             * @return The protocol family that is used for the local socket
             *         end point.
             */
            inline Socket::ProtocolFamily GetProtocolFamily(void) const {
                return static_cast<Socket::ProtocolFamily>(
                    this->bindAddress.GetAddressFamily());
            }

            /**
             * Answer the address that is sent as payload (the user address).
             *
             * @return The payload address.
             */
            inline const IPEndPoint& GetResponseAddress(void) const {
                return this->responseAddress;
            }

            /**
             * Assignment.
             *
             * @param rhs The right hand side operand.
             *
             * @return *this
             */
            DiscoveryConfig& operator =(const DiscoveryConfig& rhs);

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @return true if this object and 'rhs' are equal, 
             *         false otherwise.
             */
            bool operator ==(const DiscoveryConfig& rhs) const;

            /**
             * Test for inequality.
             *
             * @param rhs The right hand side operand.
             *
             * @return true if this object and 'rhs' are not equal, 
             *         false otherwise.
             */
            inline bool operator !=(const DiscoveryConfig& rhs) const {
                return !(*this == rhs);
            }
        
        protected:

            /** The broadcast address to send the messages to. */
            IPEndPoint bcastAddress;

            /** The address to bind the local socket to. */
            IPEndPoint bindAddress;

            /** The response address that is sent to other nodes. */
            IPEndPoint responseAddress;

        }; /* end class DiscoveryConfig */

        /**
         * This enumeration defines possibles states of the discovery service
         * as a whole.
         */
        typedef enum State_t {
            STATE_STOPPED = 0x00,
            STATE_RECEIVER_RUNNING = 0x01,
            STATE_SENDER_RUNNING = 0x02,
            STATE_RUNNING = 0x01 + 0x02
        } State;

        /** 
         * The handle that identifies a peer node. 
         *
         * No assumption should be made about the internal structure of the 
         * handle.
         */
        typedef vislib::SmartPtr<PeerNode> PeerHandle;

        /** The default port number used by the discovery service. */
        static const USHORT DEFAULT_PORT;

        /** The default request interval in milliseconds. */
        static const UINT DEFAULT_REQUEST_INTERVAL;

        /** The default number of chances to respond before disconnect. */
        static const UINT DEFAULT_RESPONSE_CHANCES;

        /** 
         * If this behaviour flag is set, the discovery service will only 
         * collect other nodes and not send alive message itself. 
         */
        static const UINT32 FLAG_OBSERVE_ONLY;

        /**
         * If this behaviour flag is set, the receiver threads will not use
         * their sockets exclusively (SetExclusiveAddrUse(false). This is a
         * potential security issue, but you might require such a behaviour
         * if multiple instances of the DiscoveryService are running on the
         * same machine and want to serve the same cluster.
         */
        static const UINT32 FLAG_SHARE_SOCKETS;

        /**
         * The maximum size of user data that can be sent via the cluster
         * discovery service in bytes.
         */
        static const SIZE_T MAX_USER_DATA = 256;

        /** 
         * The maximum length of a cluster name in characters, including the
         * trailing zero. 
         */
        static const SIZE_T MAX_NAME_LEN = MAX_USER_DATA 
            - sizeof(struct sockaddr_storage);

        /** The first message ID that can be used for a user message. */
        static const UINT32 MSG_TYPE_USER = 16;

        /**
         * Create a new instance.
         */
        DiscoveryService(void);

        /** 
         * Dtor.
         *
         * Note that the dtor will terminate the discovery.
         */
        virtual ~DiscoveryService(void);

        /**
         * Add a new ClusterDiscoveryListener to be informed about discovery
         * events.
         *
         * The caller remains owner of the memory designated by 'listener' and
         * must ensure that the object exists as long as the listener is 
         * registered.
         *
         * This method is thread-safe.
         *
         * @param listener The listener to register. This must not be NULL.
         */
        void AddListener(DiscoveryListener *listener);

        /**
         * Clear all peer nodes that have been found until now.
         *
         * This method is thread-safe.
         */
        inline void ClearPeers(void) {
            this->peerNodesCritSect.Lock();
            this->peerNodes.Clear();
            this->peerNodesCritSect.Unlock();
        }

        /**
         * Answer the number of known peer nodes. This number includes also
         * this node.
         *
         * This method is thread-safe.
         *
         * @return The number of known peer nodes.
         */
        inline SIZE_T CountPeers(void) const {
            this->peerNodesCritSect.Lock();
            SIZE_T retval = this->peerNodes.Count();
            this->peerNodesCritSect.Unlock();
            return retval;
        }

        /**
         * Answer the address the service is listening on for discovery
         * requests.
         *
         * @return The address the listening socket is bound to.
         */
        //inline const IPEndPoint& GetBindAddr(void) const {
        //    return this->bindAddr;
        //}

        /** 
         * Answer the number of chances a node gets to respond before it is
         * implicitly disconnected from the cluster.
         *
         * @return The number of chances for a node to answer.
         */
        inline UINT GetCntResponseChances(void) const {
            return this->cntResponseChances;
        }

        /**
         * Answer the source IP address 'hPeer' uses for discovery 
         * communication.
         *
         * This method is thread-safe.
         * It is safe to call this method with an illegal handle.
         *
         * @param hPeer The handle of the peer node.
         *
         * @return The IP address of the adapter that is used by 'hPeer' for 
         *         the discovery communication.
         */
        IPAddress GetDiscoveryAddress4(const PeerHandle& hPeer) const;

        /**
         * Answer the source IP address 'hPeer' uses for discovery 
         * communication. 
         *
         * This method is thread-safe.
         * It is safe to call this method with an illegal handle.
         *
         * @param hPeer The handle of the peer node.
         *
         * @return The IP address of the adapter that is used by 'hPeer' for the
         *         discovery communication.
         */
        IPAddress6 GetDiscoveryAddress6(const PeerHandle& hPeer) const;

        /**
         * Answer the configuration flags of the discovery service.
         *
         * @return The configuration flags of the discovery service.
         */
        inline UINT32 GetFlags(void) const {
            return this->flags;
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
         * Answer the interval between two discovery requests in milliseconds.
         *
         * @return The interval between two discovery  requests in milliseconds.
         */
        inline UINT GetRequestInterval(void) const {
            return this->requestInterval;
        }

        /**
         * Answer the user communication address (ID) of 'hPeer'. 
         *
         * This method is thread-safe.
         *
         * @param hPeer The handle of the peer node.
         *
         * @return The end point that has been specified by the peer node for 
         *         user communication.
         *
         * @throws IllegalParamException If 'hPeer' is not a valid handle.
         */
        inline IPEndPoint GetResponseAddress(const PeerHandle& hPeer) const {
            return (*this)[hPeer];
        }

        /**
         * Answer the state of the discovery service.
         *
         * @return The state of the discovery service.
         */
        State GetState(void) const;

        /**
         * Answer whether the discovery service will not send MSG_TYPE_IAMALIVE 
         * for being added to the peer list of other nodes.
         *
         * @return true if the node is only observing other ones, false if it 
         *         sending alive messages.
         */
        inline bool IsObserver(void) const {
            return ((this->flags & FLAG_OBSERVE_ONLY) != 0);
        }

        /**
         * Answer whether the discovery service is running. This is the case, if
         * both, the sender and the receiver thread are running.
         *
         * If the discovery service is in observer mode, only the receivers are
         * required to run in order for this method to return true.
         *
         * @return true, if the service is running, false otherwise.
         */
        bool IsRunning(void) const;

        /**
         * Answer whether the 'idx'th known peer node is this node. 
         *
         * Note: Only node 0 should answer true here. Otherwise, the system is
         * probably misconfigured.
         *
         * @param idx The index of the node to answer, which must be within 
         *            [0, CountPeers()[.
         *
         * @return true if the specified node is this node, false otherwise.
         *
         * @throws OutOfRangeException If 'idx' is not a valid node index.
         */
        bool IsSelf(const INT idx) const;

        /**
         * Answer whether the sockets of the receiver threads may be shared or 
         * not.
         *
         * @return true if the sharing flag is set, false otherwise.
         */
        inline bool IsShareSockets(void) const {
            return ((this->flags & FLAG_SHARE_SOCKETS) != 0);
        }

        /**
         * Answer whether the discovery service is stopped. This is the case, if 
         * none of the threads is running, i. e. neither the sender nor the
         * receiver thread.
         *
         * @return true, if the service is stopped, false otherwise.
         */
        inline bool IsStopped(void) const {
            return (this->GetState() == STATE_STOPPED);
        }

        /**
         * Removes, if registered, 'listener' from the list of objects informed
         * about discovery events.
         * 
         * The caller remains owner of the memory designated by 'listener'.
         *
         * This method is thread-safe.
         *
         * @param listener The listener to be removed. Nothing happens, if the
         *                 listener was not registered.
         */
        void RemoveListener(DiscoveryListener *listener);

        /**
         * Send a user-defined message to all known cluster members. The user
         * message can be an arbitrary sequence of a most MAX_USER_DATA bytes.
         *
         * You must have called Socket::Startup before you can use this method.
         *
         * @param msgType The message type identifier. This must be a 
         *                user-defined value of MSG_TYPE_USER or larger.
         * @param msgBody A pointer to the message body. This must not be NULL.
         * @param msgSize The number of valid bytes is 'msgBody'. This must be
         *                most MAX_USER_DATA. All bytes between 'msgSize' and
         *                MAX_USER_DATA will be zeroed.
         *
         * @return Zero in case of success, the number of communication trials
         *         that failed otherwise.
         *
         * @throws SocketException       If the datagram socket for sending the 
         *                               user message could not be created.
         * @throws IllegalParamException If 'msgType' is below MSG_TYPE_USER,
         *                               or 'msgBody' is a NULL pointer,
         *                               or 'msgSize' > MAX_USER_DATA.
         */
        UINT SendUserMessage(const UINT32 msgType, const void *msgBody, 
            const SIZE_T msgSize);

        /**
         * Send a user-defined message to the node that is identified with the
         * peer node handle 'hPeer'.The user message can be an arbitrary 
         * sequence of a most MAX_USER_DATA bytes.
         *
         * You must have called Socket::Startup before you can use this method.
         *
         * @param hPeer   Handle to the peer node to send the message to.
         * @param msgType The message type identifier. This must be a 
         *                user-defined value of MSG_TYPE_USER or larger.
         * @param msgBody A pointer to the message body. This must not be NULL.
         * @param msgSize The number of valid bytes is 'msgBody'. This must be
         *                most MAX_USER_DATA. All bytes between 'msgSize' and
         *                MAX_USER_DATA will be zeroed.
         *
         * @return Zero in case of success, the number of communication trials
         *         that failed otherwise.
         *
         * @throws SocketException       If the datagram socket for sending the 
         *                               user message could not be created.
         * @throws IllegalParamException If 'hPeer' is not a valid handle.
         *                               or 'msgType' is below MSG_TYPE_USER,
         *                               or 'msgBody' is a NULL pointer,
         *                               or 'msgSize' > MAX_USER_DATA.
         */
        UINT SendUserMessage(const PeerHandle& hPeer, const UINT32 msgType,
            const void *msgBody, const SIZE_T msgSize);

        /**
         * Change the interval between two discovery requests.
         *
         * This method is thread-safe by means of interlocked access to the
         * member.
         *
         * @param requestInterval The interval between two discovery  requests 
         *                        in milliseconds.
         */
        //inline void SetRequestInterval(const UINT requestInterval) {
        //    sys::Interlocked::Exchange(
        //        reinterpret_cast<INT32 *>(&this->requestInterval),
        //        static_cast<INT32>(requestInterval));
        //}

        /**
         * Start the discovery service. The service starts broadcasting requests
         * into the network and receiving the messages from other nodes. As long
         * as these threads are running, the node is regarded to be a member of
         * the specified cluster.
         *
         * @param name                     This is the name of the cluster to 
         *                                 detect. It is used to ensure that 
         *                                 nodes answering a discovery request 
         *                                 want to join the same cluster. The 
         *                                 name must have at most MAX_NAME_LEN 
         *                                 characters. If it is longer, it will 
         *                                 be truncated by the DiscoveryService.
         * @param configs
         * @param cntConfigs
         * @param flags                    A bitmask of flags configuring the 
         *                                 detail behaviour of the discovery 
         *                                 service.
         * @param cntExpectedNodes         The number of nodes expected. Until
         *                                 this number of nodes is found, an
         *                                 intensive (faster) search is 
         *                                 performed. If this number is zero,
         *                                 which is the default, no intensive
         *                                 search is performed.
         * @param requestInterval          The interval between two discovery 
         *                                 requests in milliseconds.
         * @param requestIntervalIntensive The request interval for intensive
         *                                 search.
         * @param cntResponseChances       The number of requests that another 
         *                                 node may not answer before being 
         *                                 removed from this nodes list of 
         *                                 known peers.
         *
         * @throws SystemException If the creation of one or more threads 
         *                         failed.
         * @throws std::bad_alloc  If there is not enough memory for the threads
         *                         available.
         */
        virtual void Start(const char *name, 
            const DiscoveryConfig *configs, const SIZE_T cntConfigs,
            const UINT cntExpectedNodes = 0,
            const UINT32 flags = 0,
            const UINT requestInterval = DEFAULT_REQUEST_INTERVAL,
            const UINT requestIntervalIntensive = DEFAULT_REQUEST_INTERVAL / 2,
            const UINT cntResponseChances = DEFAULT_RESPONSE_CHANCES);

        /**
         * Answer a string representation of the discovery service, which is 
         * its cluster name.
         *
         * @return A string representation.
         */
        inline StringA ToStringA(void) const {
            return this->GetName();
        }

        /**
         * Answer a string representing the peer node.
         *
         * This method is thread-safe.
         *
         * @param hPeer The handle of the peer node.
         *
         * @return A string representing the peer node.
         */
        StringA ToStringA(const PeerHandle& hPeer) const;

        /**
         * Answer a string representation of the discovery service, which is 
         * its cluster name.
         *
         * @return A string representation.
         */
        inline StringW ToStringW(void) const {
            return StringW(this->GetName());
        }

        /**
         * Answer a string representing the peer node.
         *
         * This method is thread-safe.
         *
         * @param hPeer The handle of the peer node.
         *
         * @return A string representing the peer node.
         */
        StringW ToStringW(const PeerHandle& hPeer) const;

        /** 
         * Stop the discovery service.
         *
         * This operation stops the broadcaster and the receiver thread. The
         * broadcaster will send a disconnect message before it ends.
         *
         * @param noWait If set true, the method will not block.
         *
         * @return true, if the threads have been terminated without any
         *         problem, false, if a SystemException has been thrown by 
         *         one of the threads or if the thread did not acknowledge
         *         a 'noWait' terminate.
         */
        virtual bool Stop(const bool noWait = false);

        /**
         * Answer the application defined communication address (ID) of the 
         * 'idx'th peer node.
         *
         * @param idx The index of the node to answer, which must be within 
         *            [0, CountPeers()[.
         *
         * @return The response address of the 'idx'th node.
         *
         * @throws OutOfRangeException If 'idx' is not a valid node index.
         */
        inline IPEndPoint operator [](const INT idx) const {
            this->peerNodesCritSect.Lock();
            IPEndPoint retval = this->peerNodes[idx]->responseAddress;
            this->peerNodesCritSect.Unlock();
            return retval;
        }

        /**
         * Answer the application defined communication address (ID) of the 
         * 'idx'th peer node.
         *
         * @param idx The index of the node to answer, which must be within 
         *            [0, CountPeers()[.
         *
         * @return The response address of the 'idx'th node.
         *
         * @throws OutOfRangeException If 'idx' is not a valid node index.
         */
        inline IPEndPoint operator [](const SIZE_T idx) const {
            this->peerNodesCritSect.Lock();
            IPEndPoint retval = this->peerNodes[idx]->responseAddress;
            this->peerNodesCritSect.Unlock();
            return retval;
        }

        /**
         * Answer the application defined communication address (ID) of the 
         * peer node identified by 'hPeer'.
         *
         * @param hPeer The handle of the peer node.
         *
         * @return The socket address that has been specified by the peer node
         *         for user communication.
         *
         * @throws IllegalParamException If 'hPeer' is not a valid handle for
         *                               a cluster member.
         */
        IPEndPoint operator [](const PeerHandle& hPeer) const;

    private:

        /**
         * 'SenderMessageBody' is used for MSG_TYPE_IAMHERE, MSG_TYPE_IAMALIVE 
         * and MSG_TYPE_SAYONARA in the sender thread.
         * It combines the response socket address and the cluster name in a
         * single structure.
         */
        typedef struct SenderBody_t {
            struct sockaddr_storage ResponseAddress;   //< Peer address to use.
            char Name[MAX_NAME_LEN];                   //< Name of cluster.
        } SenderMessageBody;


        /**
         * This structure is sent as message by the discovery service. Only one 
         * type of message is used as we cannot know the order and size of
         * UDP datagrams in advance.
         */
        typedef struct Message_t {
            UINT32 MagicNumber;						    //< Always MAGIC_NUMBER.
            UINT32 MsgType;							    //< The type identifier.
            // Note: 'magicNumber' and 'msgType' can be 32 bit now, because 
            // struct sockaddr_storage must be 64 bit aligned any way.
            union {
                SenderMessageBody SenderBody;           //< I am here messages.
                struct sockaddr_storage ResponseAddress;//< Response peer addr.
                BYTE UserData[MAX_USER_DATA];		    //< User defined data.
            };
        } Message;


        /**
         * This Runnable receives discovery requests from other nodes. User
         * messages are also received by this thread and directed to all
         * registered listeners of the ClusterDiscoveryService.
         */
        class Receiver : public vislib::sys::Runnable {

        public:

            /** Create a new instance. */
            Receiver(void);

            /** Dtor. */
            virtual ~Receiver(void);

            /**
             * Answers the discovery requests.
             *
             * @param dcfg The configuration to work for. The caller must 
             *             ensure that the object exists long enough - the 
             *             object does not take ownership of 'config'.
             *
             * @return 0, if the work was successfully finished, an error code
             *         otherwise.
             */
            virtual DWORD Run(void *dcfg);

            /**
             * Ask the thread to terminate.
             *
             * @return true.
             */
            virtual bool Terminate(void);

        private:

            /** Flag for terminating the thread safely. */
            INT32 isRunning;

            /** The socket used for receiving messages. */
            Socket socket;

        }; /* end class Receiver */


        /**
         * This class stores a per-adapter discovery configuration that is 
         * enhanced with internal data for running the receiver threads. Those
         * additional data are only added internally such that the extended
         * data cannot accessed by the user from outside the DiscoveryService.
         */
        class DiscoveryConfigEx : public DiscoveryConfig {

        public:

            DiscoveryConfigEx(void);

            DiscoveryConfigEx(const DiscoveryConfig& config, 
                DiscoveryService *cds);

            DiscoveryConfigEx(const DiscoveryConfigEx& rhs);

            virtual ~DiscoveryConfigEx(void);

            inline DiscoveryService& GetDiscoveryService(void) {
                ASSERT(this->cds != NULL);
                return *(this->cds);
            }

            inline sys::Thread& GetRecvThread(void) {
                return this->recvThread;
            }

            inline const sys::Thread& GetRecvThread(void) const {
                return this->recvThread;
            }

            /**
             * Set the response address of this configuration in 'message' and
             * send the message using the 'socketSend' socket. The messages are
             * directed to the broadcast address.
             *
             * The respons address is set if the message type is one of the 
             * following: MSG_TYPE_IAMHERE, MSG_TYPE_IAMALIVE, MSG_TYPE_SAYONARA
             *
             * Otherwise, the message is sent without customisation.
             *
             * @param message The message to send.
             */
            void SendCustomisedMessage(Message& message);

            void SendMessageTo(Message& message, const IPEndPoint& to);

            /**
             * Assignment.
             *
             * @param rhs The right hand side operand.
             *
             * @return *this
             */
            DiscoveryConfigEx& operator =(const DiscoveryConfigEx& rhs);

            /**
             * Test for equality.
             *
             * @param rhs The right hand side operand.
             *
             * @return true if this object and 'rhs' are equal, 
             *         false otherwise.
             */
            bool operator ==(const DiscoveryConfigEx& rhs) const;

            /**
             * Test for inequality.
             *
             * @param rhs The right hand side operand.
             *
             * @return true if this object and 'rhs' are not equal, 
             *         false otherwise.
             */
            inline bool operator !=(const DiscoveryConfigEx& rhs) const {
                return !(*this == rhs);
            }

        private:

            /* Superclass typedef. */
            typedef DiscoveryConfig Super;

            /** Back reference to the discovery service. */
            DiscoveryService *cds;

            /** 
             * The receive worker thread that is responsible for answering
             * requests for this configuration.
             */
            sys::RunnableThread<Receiver> recvThread;

            /** The socket for the message sender thread. */
            Socket socketSend;

            /** The socket used for the user message. */
            Socket socketUserMsg;

        }; /* end class DiscoveryConfigEx */


        /**
         * This Runnable is the worker that broadcasts the discovery requests of
         * for a specific DiscoveryService. 
         *
         * Each DiscoveryService has one sender thread, even if it uses multiple
         * adapters. In this case, the sender will process all adapters 
         * sequentially.
         */
        class Sender : public vislib::sys::Runnable {

        public:
            
            /**
             * Create a new instance that is working for a discovery service.
             */
            Sender(void);

            /** Dtor. */
            virtual ~Sender(void);

            /**
             * Performs the discovery.
             *
             * @param cds The DiscoveryService that determines the communication
             *            parameters and receives the peer nodes that have been 
             *            detected.
             *
             * @return 0, if the work was successfully finished, an error code
             *         otherwise.
             */
            virtual DWORD Run(void *cds);

            /**
             * Ask the thread to terminate.
             *
             * This operation resets the 'isRunning' flags which causes the 
             * thread to leave at latest after the send timeout has elapsed.
             *
             * @return true.
             */
            virtual bool Terminate(void);

        private:

            /** Flag for terminating the thread safely. */
            INT32 isRunning;

        }; /* end class Sender */


        /** An array for storing the per-adapter configurations. */
        typedef Array<DiscoveryConfigEx> ConfigList;

        /** A shortcut to the listener list type. */
        typedef SingleLinkedList<DiscoveryListener *, 
            vislib::sys::CriticalSection> ListenerList;

        /** An array for storing the known peer nodes. */
        typedef Array<PeerHandle> PeerNodeList;

        /**
         * Add a peer to the list of known peer nodes of the discovery
         * service. If node is already known, the response chance counter 
         * is reset to its original value.
         *
         * This method also fires the node found event by calling OnNodeFound
         * on all registered DiscoveryListeners.
         *
         * This method is thread-safe.
         *
         * @param discoveryAddr   The socket address the discovery service of 
         *                        the peer node sent the message. Note, that 
         *                        the port is automatically corrected to the 
         *                        correct service port and must not be set by 
         *                        the caller.
         * @param discoverySource The configuration of the receiver that 
         *                        discovered the new node.
         * @param responseAddr    The user communication address the peer node
         *                        reported.
         */
        void addPeerNode(const IPEndPoint& discoveryAddr, 
            DiscoveryConfigEx *discoverySource,
            const IPEndPoint& responseAddr);

         /**
          * Inform all registered DiscoveryListeners of the discovery
          * service about a a user message that we received.
          *
          * The method will lookup the response address that is passed to 
          * the registered listeners itself. 'sender' must therefore be the 
          * UDP address of the discovery service. Only the IPAddress of 
          * 'sender' is used to identify the peer node, the port is discarded. 
          * The method accepts a socket address instead of an IPAddress just 
          * for convenience.
          *
          * If the sender node was not found in the peer node list, no event
          * is fired!
          *
          * This method is thread-safe.
          *
          * @param sender          The socket address of the message sender.
          * @param discoverySource The configuration of the receiver that 
          *                        received the user message.
          * @param msgType         The message type.
          * @param msgBody         The body of the message.
          */
         void fireUserMessage(const IPEndPoint& sender, 
             DiscoveryConfigEx *discoverySource, const UINT32 msgType, 
             const BYTE *msgBody); 

        /**
         * Answer whether 'hPeer' is a valid peer node handle.
         *
         * @param hPeer Handle to a peer node.
         *
         * @return true, if 'hPeer' is valid, false otherwise.
         */
        bool isValidPeerHandle(const PeerHandle& hPeer) const;

        /**
         * Answer the index of the peer node that reported the user 
         * communication address 'addr'. If no such node exists, -1 
         * is returned.
         *
         * This method is NOT thread-safe.
         *
         * @param addr The discovery address to lookup.
         *
         * @return The index of the peer node or -1, if not found.
         */
        INT_PTR peerFromResponseAddress(const IPEndPoint& addr) const;

        /**
         * Answer the index of the peer node that runs its discovery 
         * service on 'addr'. If no such node exists, -1 is returned.
         *
         * Only the IP address is taken into account, not the port. It is
         * therefore safe to use the UDP sender address for 'addr'.
         *
         * This method is NOT thread-safe.
         *
         * @param addr The discovery address to lookup.
         *
         * @return The index of the peer node or -1, if not found.
         */
        INT_PTR peerFromDiscoveryAddr(const IPEndPoint& addr) const;

        /**
         * Prepares the list of known peer nodes for a new request. 
         *
         * To do so, the 'cntResponseChances' of each node is checked whether it
         * is zero. If this condition holds true, the node is removed. 
         * Otherwise, the 'cntResponseChances' is decremented.
         *
         * This method also fires the node lost event by calling OnNodeLost
         * on all registered ClusterDiscoveryListeners.
         *
         * This method is thread-safe.
         */
        void prepareRequest(void);

        /**
         * This method prepares sending a user message.
         *
         * The following steps are taken:
         *
         * All user input, i. e. 'msgType', 'msgBody' and 'msgSize' is validated
         * and an IllegalParamException is thrown, if 'msgType' is not in the
         * user message range, if 'msgBody' is a NULL pointer of 'msgSize' is 
         * too large.
         *
         * If all user input is valid, the datagram to be sent is written to
         * 'outMsg'.
         *
         * @param outMsg  The Message structure receiving the datagram.
         * @param msgType The message type identifier. This must be a 
         *                user-defined value of MSG_TYPE_USER or larger.
         * @param msgBody A pointer to the message body. This must not be NULL.
         * @param msgSize The number of valid bytes is 'msgBody'. This must be
         *                most MAX_USER_DATA. All bytes between 'msgSize' and
         *                MAX_USER_DATA will be zeroed.
         *
         * @throws IllegalParamException If 'msgType' is below MSG_TYPE_USER,
         *                               or 'msgBody' is a NULL pointer,
         *                               or 'msgSize' > MAX_USER_DATA.
         */
        void prepareUserMessage(Message& outMsg, const UINT32 msgType,
            const void *msgBody, const SIZE_T msgSize);

        /**
         * Remove the peer node having the user communication address 'address'.
         * If no such node exists, nothing will happen.
         *
         * This method also fires the node lost event sigaling an explicit
         * remove of the node.
         *
         * @param address The socket address that the peer node reported as its
         *                user communication port.
         */
        void removePeerNode(const IPEndPoint& address);

        /** The magic number at the begin of each message. */
        static const UINT32 MAGIC_NUMBER;

        /** Message type ID of a repeated discovery request. */
        static const UINT32 MSG_TYPE_IAMALIVE = 2;

        /** Message type ID of an initial discovery request. */
        static const UINT32 MSG_TYPE_IAMHERE = 1;

        /** Message type ID of the explicit disconnect notification. */
        static const UINT32 MSG_TYPE_SAYONARA = 3;

        /** 
         * The number of expected nodes. Until this number is reached, the
         * discovery service performs an intensive search for peer nodes.
         */
        UINT cntExpectedNodes;

        /** 
         * The number of chances a node gets to respond before it is implicitly 
         * disconnected from the cluster.
         */
        UINT cntResponseChances;

        /** Boolean flags that specialise the behaviour of the object. */
        UINT flags;

        /** The time in milliseconds between two discovery requests. */
        UINT requestInterval;

        /** The interval for intensive search in milliseconds. */
        UINT requestIntervalIntensive;

        /** The per-adapter configurations. */
        ConfigList configs;

        /** This list holds the objects to be informed about new nodes. */
        ListenerList listeners;

        /** The name of the cluster this discovery service should form. */
        StringA name;

        /** 
         * The thread sending the alive beacons of this node to the broadcast
         * addresses specified in the discovery service configurations.
         */
        sys::RunnableThread<Sender> senderThread;

        /** This array holds the peer nodes. */
        PeerNodeList peerNodes;

        /** Critical section for protecting 'peerNodes'. */
        mutable sys::CriticalSection peerNodesCritSect;

        /* Allow access to private methods. */
        friend class Receiver;
    };


} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CLUSTERDISCOVERYSERVICE_H_INCLUDED */
