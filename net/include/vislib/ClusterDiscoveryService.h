/*
 * ClusterDiscoveryService.h
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


#include "vislib/IPAddress.h"   // Must be included at begin!
#include "vislib/Array.h"
#include "vislib/CriticalSection.h"
#include "vislib/Interlocked.h"
#include "vislib/IPAddress6.h"
#include "vislib/RunnableThread.h"
#include "vislib/SingleLinkedList.h"
#include "vislib/Socket.h"
#include "vislib/SmartPtr.h"
#include "vislib/String.h"
#include "vislib/types.h"



namespace vislib {
namespace net {

    /* Forward declarations. */
    class ClusterDiscoveryListener;


    /**
     * This class implements a method for discovering other computers in a
     * network via UDP broadcasts. 
     *
     * The user specifies a name as identifier of the cluster to be searched
     * and the object creates an array of all nodes that respond to a request
     * whether they are also members of this cluster. The object also anwers
     * requests of other nodes.
     *
     * Remarks concerning IPv6 support: The broadcast address determines 
     * whether the cluster discovery service is running in IPv4 or IPv6 mode.
     * The response address that is used as node identifier is independent from
     * the mode the discovery service is running in.
     */
    class ClusterDiscoveryService {

    private:

        /** 
         * This structure is used to identify a peer node that has been found
         * during the discovery process.
         */
        typedef struct PeerNode_t {
            IPEndPoint address;             // User communication address (ID).
            IPEndPoint discoveryAddr;       // Discovery service address.
            UINT cntResponseChances;        // Implicit disconnect detector.

            inline bool operator ==(const PeerNode_t& rhs) const {
                return (this->address == rhs.address);
            }
        } PeerNode;

    public:

        typedef vislib::SmartPtr<PeerNode> PeerHandle;

        /** The default port number used by the discovery service. */
        static const USHORT DEFAULT_PORT;

        /** The default request interval in milliseconds. */
        static const UINT DEFAULT_REQUEST_INTERVAL;

        /** The default number of chances to respond before disconnect. */
        static const UINT DEFAULT_RESPONSE_CHANCES;

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
        static const UINT32 MSG_TYPE_USER;

        /**
         * Create a new instance.
         *
         * @param name                This is the name of the cluster to 
         *                            detect. It is used to ensure that nodes
         *                            answering a discovery request want to 
         *                            join the same cluster. The name must have 
         *                            at most MAX_NAME_LEN characters.
         * @param responseAddr        This is the "call back address" of the 
         *                            current node, on which user-defined 
         *                            communication should be initiated. The 
         *                            ClusterDiscoveryService does not use this 
         *                            address itself, but justcommunicates it to
         *                            all other nodes which then can use it. 
         *                            These addresses should uniquely identify
         *                            each process in the cluster, i. e. no node
         *                            should specify the same 'responseAddr' as
         *                            some other does.
         * @param bcastAddr           The broadcast address of the network. All
         *                            requests will be sent to this address. The
         *                            destination port of messages is derived 
         *                            from 'bindAddr'. 
         *                            You can use the NetworkInformation to 
         *                            obtain the broadcast address of your 
         *                            subnet.
         * @param bindPort            The port to bind the receiver thread to.
         *                            All discovery requests must be directed to
         *                            this port.
         * @param isObserver          If this flag is set, the discovery service 
         *                            will only discover nodes, but not request
         *                            membership in the cluster.
         * @param requestInterval     The interval between two discovery 
         *                            requests in milliseconds.
         * @param cntResponseChances  The number of requests that another node 
         *                            may not answer before being removed from
         *                            this nodes list of known peers.
         */
        ClusterDiscoveryService(const StringA& name, 
            const IPEndPoint& responseAddr, 
            const IPAddress& bcastAddr,
            const USHORT bindPort = DEFAULT_PORT,
            const bool discoveryOnly = false,
            const UINT requestInterval = DEFAULT_REQUEST_INTERVAL,
            const UINT cntResponseChances = DEFAULT_RESPONSE_CHANCES);

        /**
         * Create a new instance.
         *
         * @param name                This is the name of the cluster to 
         *                            detect. It is used to ensure that nodes
         *                            answering a discovery request want to 
         *                            join the same cluster. The name must have 
         *                            at most MAX_NAME_LEN characters.
         * @param responseAddr        This is the "call back address" of the 
         *                            current node, on which user-defined 
         *                            communication should be initiated. The 
         *                            ClusterDiscoveryService does not use this 
         *                            address itself, but justcommunicates it to
         *                            all other nodes which then can use it. 
         *                            These addresses should uniquely identify
         *                            each process in the cluster, i. e. no node
         *                            should specify the same 'responseAddr' as
         *                            some other does.
         * @param bcastAddr           The broadcast address of the network. All
         *                            requests will be sent to this address. The
         *                            destination port of messages is derived 
         *                            from 'bindAddr'. 
         *                            You can use the NetworkInformation to 
         *                            obtain the broadcast address of your 
         *                            subnet.
         * @param bindPort            The port to bind the receiver thread to.
         *                            All discovery requests must be directed to
         *                            this port.
         * @param isObserver          If this flag is set, the discovery service 
         *                            will only discover nodes, but not request
         *                            membership in the cluster.
         * @param requestInterval     The interval between two discovery 
         *                            requests in milliseconds.
         * @param cntResponseChances  The number of requests that another node 
         *                            may not answer before being removed from
         *                            this nodes list of known peers.
         */
        ClusterDiscoveryService(const StringA& name, 
            const IPEndPoint& responseAddr, 
            const IPAddress6& bcastAddr,
            const USHORT bindPort = DEFAULT_PORT,
            const bool discoveryOnly = false,
            const UINT requestInterval = DEFAULT_REQUEST_INTERVAL,
            const UINT cntResponseChances = DEFAULT_RESPONSE_CHANCES);

        /** 
         * Dtor.
         *
         * Note that the dtor will terminate the discovery.
         */
        virtual ~ClusterDiscoveryService(void);

        /**
         * Add a new ClusterDiscoveryListener to be informed about discovery
         * events. The caller remains owner of the memory designated by 
         * 'listener' and must ensure that the object exists as long as the
         * listener is registered.
         *
         * @param listener The listener to register. This must not be NULL.
         */
        void AddListener(ClusterDiscoveryListener *listener);

        /**
         * Clear all peer nodes that have been found until now.
         */
        inline void ClearPeers(void) {
            this->critSect.Lock();
            this->peerNodes.Clear();
            this->critSect.Unlock();
        }

        /**
         * Answer the number of known peer nodes. This number includes also
         * this node.
         *
         * @return The number of known peer nodes.
         */
        inline SIZE_T CountPeers(void) const {
            this->critSect.Lock();
            SIZE_T retval = this->peerNodes.Count();
            this->critSect.Unlock();
            return retval;
        }

        /**
         * Answer the address the service is listening on for discovery
         * requests.
         *
         * @return The address the listening socket is bound to.
         */
        inline const IPEndPoint& GetBindAddr(void) const {
            return this->bindAddr;
        }

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
         * Answer the source IP address 'hPeer' uses for discovery communication. 
         *
         * @param hPeer The handle of the peer node.
         *
         * @return The IP address of the adapter that is used by 'hPeer' for the
         *         discovery communication.
         *
         * @throws IllegalParamException If 'hPeer' is not a valid handle.
         */
        IPAddress GetDiscoveryAddress4(const PeerHandle& hPeer) const;

        /**
         * Answer the source IP address 'hPeer' uses for discovery communication. 
         *
         * @param hPeer The handle of the peer node.
         *
         * @return The IP address of the adapter that is used by 'hPeer' for the
         *         discovery communication.
         *
         * @throws IllegalParamException If 'hPeer' is not a valid handle.
         */
        IPAddress6 GetDiscoveryAddress6(const PeerHandle& hPeer) const;

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
         * Answer the call back socket address that is sent to peer nodes 
         * when they are discovered. This address can be used to establish a
         * connection to our node in a application defined manner.
         *
         * @return The address sent as response.
         */
        inline const IPEndPoint& GetResponseAddr(void) const {
            return this->responseAddr;
        }

        /**
         * Answer the user communication address of 'hPeer'. 
         *
         * @param hPeer The handle of the peer node.
         *
         * @return The socket address that has been specified by the peer node
         *         for user communication.
         *
         * @throws IllegalParamException If 'hPeer' is not a valid handle.
         */
        inline IPEndPoint GetUserComAddress(const PeerHandle& hPeer) const {
            return (*this)[hPeer];
        }

        /**
         * Answer whether the discovery service will not send MSG_TYPE_IAMALIVE 
         * for being added to the peer list of other nodes.
         *
         * @return true if the node is only observing other ones, false if it 
         *         sending alive messages.
         */
        inline bool IsObserver(void) const {
            return this->isObserver;
        }

        /**
         * Answer whether the discovery service is running. This is the case, if
         * both, the sender and the receiver thread are running.
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
        inline bool IsSelf(const INT idx) const {
            this->critSect.Lock();
            bool retval = (this->peerNodes[idx]->address == this->responseAddr);
            this->critSect.Unlock();
            return retval;
        }

        /**
         * Answer whether the discovery service is stopped. This is the case, if 
         * none of the threads is running, i. e. neither the sender nor the
         * receiver thread.
         *
         * @return true, if the service is stopped, false otherwise.
         */
        bool IsStopped(void) const;

        /**
         * Removes, if registered, 'listener' from the list of objects informed
         * about discovery events. The caller remains owner of the memory 
         * designated by 'listener'.
         *
         * @param listener The listener to be removed. Nothing happens, if the
         *                 listener was not registered.
         */
        void RemoveListener(ClusterDiscoveryListener *listener);

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
        inline void SetRequestInterval(const UINT requestInterval) {
            sys::Interlocked::Exchange(
                reinterpret_cast<INT32 *>(&this->requestInterval),
                static_cast<INT32>(requestInterval));
        }

        /**
         * Start the discovery service. The service starts broadcasting requests
         * into the network and receiving the messages from other nodes. As long
         * as these threads are running, the node is regarded to be a member of
         * the specified cluster.
         *
         * @throws SystemException If the creation of one or more threads 
         *                         failed.
         * @throws std::bad_alloc  If there is not enough memory for the threads
         *                         available.
         */
        virtual void Start(void);

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
         * Answer the application defined communication address of the 'idx'th
         * peer node.
         *
         * @param idx The index of the node to answer, which must be within 
         *            [0, CountPeers()[.
         *
         * @return The response address of the 'idx'th node.
         *
         * @throws OutOfRangeException If 'idx' is not a valid node index.
         */
        inline IPEndPoint operator [](const INT idx) const {
            this->critSect.Lock();
            IPEndPoint retval = this->peerNodes[idx]->address;
            this->critSect.Unlock();
            return retval;
        }

        /**
         * Answer the application defined communication address of the 'idx'th
         * peer node.
         *
         * @param idx The index of the node to answer, which must be within 
         *            [0, CountPeers()[.
         *
         * @return The response address of the 'idx'th node.
         *
         * @throws OutOfRangeException If 'idx' is not a valid node index.
         */
        inline IPEndPoint operator [](const SIZE_T idx) const {
            this->critSect.Lock();
            IPEndPoint retval = this->peerNodes[idx]->address;
            this->critSect.Unlock();
            return retval;
        }

        /**
         * Answer the user communication address of 'hPeer'. 
         *
         * @param hPeer The handle of the peer node.
         *
         * @return The socket address that has been specified by the peer node
         *         for user communication.
         *
         * @throws IllegalParamException If 'hPeer' is not a valid handle.
         */
        IPEndPoint operator [](const PeerHandle& hPeer) const;

    protected:

        /**
         * This Runnable is the worker that broadcasts the discovery requests of
         * for a specific ClusterDiscoveryService.
         */
        class Sender : public vislib::sys::Runnable {

        public:
            
            /**
             * Create a new instance that is working for 'cds'.
             */
            Sender(void);

            /** Dtor. */
            virtual ~Sender(void);

            /**
             * Performs the discovery.
             *
             * @param discSvc The ClusterDiscoveryService that determines the 
             *                communication parameters and receives the peer nodes
             *                that have been detected.
             *
             * @return 0, if the work was successfully finished, an error code
             *         otherwise.
             */
            virtual DWORD Run(void *discSvc);

            /**
             * Ask the thread to terminate.
             *
             * @return true.
             */
            virtual bool Terminate(void);

        private:

            /** Flag for terminating the thread safely. */
            bool isRunning;

        }; /* end class Sender */


        /**
         * This Runnable receives discovery requests from other nodes. User
         * messages are also received by this thread and directed to all
         * registered listeners of the ClusterDiscoveryService.
         */
        class Receiver : public vislib::sys::Runnable {

        public:

            /**
             * Create a new instance answering discovery requests directed to
             * 'cds'.
             */
            Receiver(void);

            /** Dtor. */
            virtual ~Receiver(void);

            /**
             * Answers the discovery requests.
             *
             * @param discSvc The ClusterDiscoveryService that determines the 
             *                communication parameters and receives the peer nodes
             *                that have been detected.
             *
             * @return 0, if the work was successfully finished, an error code
             *         otherwise.
             */
            virtual DWORD Run(void *discSvc);

            /**
             * Ask the thread to terminate.
             *
             * @return true.
             */
            virtual bool Terminate(void);

        private:

            /** Flag for terminating the thread safely. */
            bool isRunning;

            /** The socket used for the broadcast. */
            Socket socket;

        }; /* end class Receiver */

        /* 
         * 'SenderMessageBody' is used for MSG_TYPE_DISCOVERY_REQUEST and
         * MSG_TYPE_SAYONARA in the sender thread.
         */
        typedef struct SenderBody_t {
            struct sockaddr_storage sockAddr;   // Peer address to use.
            char name[MAX_NAME_LEN];		    // Name of cluster.
        } SenderMessageBody;

        /**
         * This structure is sent as message by the discovery service. Only one 
         * type of message is used as we cannot know the order and size of
         * UDP datagrams in advance.
         */
        typedef struct Message_t {
            UINT32 magicNumber;						    // Must be MAGIC_NUMBER.
            UINT32 msgType;							    // The type identifier.
            // Note: 'magicNumber' and 'msgType' can be 32 bit now, because 
            // struct sockaddr_storage must be 64 bit aligned any way.
            union {
                SenderMessageBody senderBody;           // I am here messages.
                struct sockaddr_storage responseAddr;   // Resonse peer address.
                BYTE userData[MAX_USER_DATA];		    // User defined data.
            };
        } Message;

        /** A shortcut to the listener list type. */
        typedef SingleLinkedList<ClusterDiscoveryListener *> ListenerList;

        /** Such an array is used for storing the known peer nodes. */
        typedef Array<PeerHandle> PeerNodeList;

        /**
         * Add a peer to the list of known peer nodes. If node is already known,
         * the response chance counter is reset to its original value.
         *
         * This method also fires the node found event by calling OnNodeFound
         * on all registered ClusterDiscoveryListeners.
         *
         * This method is thread-safe.
         *
         * @param discoveryAddr The socket address the discovery service of the
         *                      peer node sent the message. Note, that the 
         *                      port is automatically corrected to the correct
         *                      service port and must not be set by the caller.
         * @param address       The user communication address the peer node 
         *                      reported.
         */
        void addPeerNode(const IPEndPoint& discoveryAddr, 
            const IPEndPoint& address);

        /**
         * Inform all registered ClusterDiscoveryListeners about a a user 
         * message that we received.
         *
         * The method will lookup the response address that is passed to the
         * registered listeners itself. 'sender' must therefore be the UDP
         * address of the discovery service. Only the IPAddress of 'sender'
         * is used to identify the peer node, the port is discarded. The 
         * method accepts a socket address instead of an IPAddress just for
         * convenience.
         *
         * If the sender node was not found in the peer node list, no event
         * is fired!
         *
         * This method is thread-safe.
         *
         * @param sender  The socket address of the message sender.
         * @param msgType The message type.
         * @param msgBody The body of the message.
         */
        void fireUserMessage(const IPEndPoint& sender, const UINT32 msgType, 
            const BYTE *msgBody) const;

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
        INT_PTR peerFromAddress(const IPEndPoint& addr) const;

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
         * This method prepares sending a user message. The following 
         * steps are taken.
         *
         * All user input, i. e. 'msgType', 'msgBody' and 'msgSize' is validated
         * and an IllegalParamException is thrown, if 'msgType' is not in the
         * user message range, if 'msgBody' is a NULL pointer of 'msgSize' is 
         * too large.
         *
         * If all user input is valid, the datagram to be sent is written to
         * 'outMsg'.
         *
         * It is further checked, whether 'this->userMsgSocket' is already 
         * valid. If not, the socket is created.
         *
         * @param outMsg  The Message structure receiving the datagram.
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
        static const UINT32 MSG_TYPE_IAMALIVE;

        /** Message type ID of an initial discovery request. */
        static const UINT32 MSG_TYPE_IAMHERE;

        /** Message type ID of the explicit disconnect notification. */
        static const UINT32 MSG_TYPE_SAYONARA;

        /** This is the broadcast address to send requests to. */
        IPEndPoint bcastAddr;

        /** The address that the response thread binds to. */
        IPEndPoint bindAddr;

        /** The address we send in a response message. */
        IPEndPoint responseAddr;

        /**
         * Critical section protecting access to the 'peerNodes' array and the
         * 'listeners' list.
         */
        mutable sys::CriticalSection critSect;

        /** 
         * The number of chances a node gets to respond before it is implicitly 
         * disconnected from the cluster.
         */
        UINT cntResponseChances;

        /** The time in milliseconds between two discovery requests. */
        UINT requestInterval;

        /**
         * If this flag is set, the discovery service will not send 
         * MSG_TYPE_IAMALIVE for being added to the peer list of other nodes.
         */
        bool isObserver;

        /** This list holds the objects to be informed about new nodes. */
        ListenerList listeners;

        /** The name of the cluster this discovery service should form. */
        StringA name;

        /** The thread receiving discovery requests. */
        sys::RunnableThread<Receiver> receiverThread;

        /** The thread performing the node discovery. */
        sys::RunnableThread<Sender> senderThread;

        /** This array holds the peer nodes. */
        PeerNodeList peerNodes;

        /** The socket used for sending user messages. */
        Socket userMsgSocket;

        /* Allow threads to access protected methods. */
        friend class Receiver;
        friend class Sender;
    };


} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CLUSTERDISCOVERYSERVICE_H_INCLUDED */
