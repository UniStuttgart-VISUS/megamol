/*
 * DiscoveryService.cpp
 *
 * Copyright (C) 2006 -2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/DiscoveryService.h"

#include "vislib/assert.h"
#include "vislib/DiscoveryListener.h"
#include "vislib/IllegalParamException.h"
#include "vislib/NetworkInformation.h"
#include "vislib/SocketException.h"
#include "vislib/Trace.h"


////////////////////////////////////////////////////////////////////////////////
// Begin class PeerNode

/*
 * vislib::net::cluster::DiscoveryService::PeerNode::PeerNode
 */
vislib::net::cluster::DiscoveryService::PeerNode::PeerNode(void) 
        : cntResponseChances(1), discoverySource(NULL) {
}


/*
 * vislib::net::cluster::DiscoveryService::PeerNode::PeerNode
 */
vislib::net::cluster::DiscoveryService::PeerNode::PeerNode(
        const IPEndPoint& discoveryAddress,  
        const IPEndPoint& responseAddress,
        const UINT cntResponseChances, 
        DiscoveryConfigEx *discoverySource)
        : cntResponseChances(cntResponseChances), 
        discoveryAddress(discoveryAddress),
        discoverySource(discoverySource),
        responseAddress(responseAddress) {
    ASSERT(this->discoverySource != NULL);

    // Fix the port of the discovery address, as we get the client socket 
    // address as input, which is normally not bound to the port of the 
    // broadcast address we use.
    this->discoveryAddress.SetPort(
        this->discoverySource->GetBcastAddress().GetPort());

    VLTRACE(Trace::LEVEL_VL_VERBOSE, "New peer node %s, discovered via %s.\n",
        this->responseAddress.ToStringA().PeekBuffer(),
        this->discoveryAddress.ToStringA().PeekBuffer());
}


/*
 * vislib::net::cluster::DiscoveryService::PeerNode::operator =
 */
vislib::net::cluster::DiscoveryService::PeerNode& 
vislib::net::cluster::DiscoveryService::PeerNode::operator =(
        const PeerNode& rhs) {
    if (this != &rhs) {
        this->cntResponseChances = rhs.cntResponseChances;
        this->discoveryAddress = rhs.discoveryAddress;
        this->discoverySource = rhs.discoverySource;
        this->responseAddress = rhs.responseAddress;
    }

    return *this;
}


/*
 * vislib::net::cluster::DiscoveryService::PeerNode::DecrementResponseChances
 */
bool vislib::net::cluster::DiscoveryService::PeerNode::decrementResponseChances(
        void) {
    if (this->cntResponseChances > 0) {
        this->cntResponseChances--;
    }
    
    return (this->cntResponseChances != 0);
}


/*
 * vislib::net::cluster::DiscoveryService::PeerNode::Invalidate
 */
void vislib::net::cluster::DiscoveryService::PeerNode::invalidate(void) {
    struct sockaddr invalidAddr;
    reinterpret_cast<int&>(invalidAddr) = 0;
    // Note: The invalid address can either be IPv4 or IPv6 as the
    // end point is agnostic and we do not need an invalidated address
    // any more.

    this->responseAddress = invalidAddr;
}


/*
 * vislib::net::cluster::DiscoveryService::PeerNode::IsValid
 */
bool vislib::net::cluster::DiscoveryService::PeerNode::isValid(void) const {
    return !this->responseAddress.GetIPAddress6().IsUnspecified();
}


// End class PeerNode
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Begin class DiscoveryConfig


/*
 * vislib::net::cluster::DiscoveryService::DiscoveryConfig::DiscoveryConfig
 */
vislib::net::cluster::DiscoveryService::DiscoveryConfig::DiscoveryConfig(void) {
    // TODO: This is not good
    NetworkInformation::Adapter adapter
        = NetworkInformation::AdapterInformation(0);
    
    this->responseAddress = IPEndPoint(adapter.Address());
    this->bindAddress = IPEndPoint(adapter.Address(), DEFAULT_PORT);
    this->bcastAddress = IPEndPoint(adapter.BroadcastAddress(), DEFAULT_PORT);
}

/*
 * vislib::net::cluster::DiscoveryService::DiscoveryConfig::DiscoveryConfig
 */
vislib::net::cluster::DiscoveryService::DiscoveryConfig::DiscoveryConfig(
        const vislib::net::IPEndPoint &responseAddress, 
        const vislib::net::IPAddress &bindAddress, 
        const vislib::net::IPAddress &bcastAddress, 
        const USHORT bindPort) 
        : bcastAddress(bcastAddress, bindPort),
        bindAddress(bindAddress, bindPort),
        responseAddress(responseAddress) {
}


/*
 * vislib::net::cluster::DiscoveryService::DiscoveryConfig::DiscoveryConfig
 */
vislib::net::cluster::DiscoveryService::DiscoveryConfig::DiscoveryConfig(
        const vislib::net::IPEndPoint &responseAddress, 
        const vislib::net::IPAddress6 &bindAddress, 
        const vislib::net::IPAddress6 &bcastAddress, 
        const USHORT bindPort) 
        : bcastAddress(bcastAddress, bindPort),
        bindAddress(bindAddress, bindPort),
        responseAddress(responseAddress) {
}


/*
 * vislib::net::cluster::DiscoveryService::DiscoveryConfig::DiscoveryConfig
 */
vislib::net::cluster::DiscoveryService::DiscoveryConfig::DiscoveryConfig(
        const IPEndPoint& responseAddress, 
        const IPAddress& bindAddress, 
        const USHORT bindPort)
        : bindAddress(bindAddress, bindPort),
        responseAddress(responseAddress) {
    ASSERT(this->bcastAddress.GetPort() == 0);  // This is our guard!

    // TODO: This is not yet IPv6-compatible
    UINT cntAdapters = NetworkInformation::AdapterCount();
    for (UINT i = 0; i < cntAdapters; i++) {
        const NetworkInformation::Adapter& a 
            = NetworkInformation::AdapterInformation(i);
        IPAddress aNet = a.Address() & a.SubnetMask();
        IPAddress bNet = bindAddress & a.SubnetMask();

        if (a.SubnetMask() != IPAddress::ANY && aNet == bNet) {
            this->bcastAddress.SetIPAddress(a.BroadcastAddress());
            this->bcastAddress.SetPort(bindPort);
        }
    }

    if (this->bcastAddress.GetPort() == 0) {
        this->~DiscoveryConfig();
        throw IllegalParamException("bindAddress", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::cluster::DiscoveryService::DiscoveryConfig::DiscoveryConfig
 */
//vislib::net::cluster::DiscoveryService::DiscoveryConfig::DiscoveryConfig(
//            const IPEndPoint& responseAddress, 
//            const IPAddress6& bindAddress, 
//            const USHORT bindPort)
//        : bindAddress(bindAddress, bindPort),
//        responseAddress(bcastAddress) {
//    // TODO
//}


/*
 * vislib::net::cluster::DiscoveryService::DiscoveryConfig::DiscoveryConfig
 */
vislib::net::cluster::DiscoveryService::DiscoveryConfig::DiscoveryConfig(
        const DiscoveryConfig& rhs) {
    *this = rhs;
}


/*
 * vislib::net::cluster::DiscoveryService::DiscoveryConfig::~DiscoveryConfig
 */
vislib::net::cluster::DiscoveryService::DiscoveryConfig::~DiscoveryConfig(
        void) {
}


/*
 * vislib::net::cluster::DiscoveryService::DiscoveryConfig::operator =
 */
vislib::net::cluster::DiscoveryService::DiscoveryConfig& 
vislib::net::cluster::DiscoveryService::DiscoveryConfig::operator =(
        const DiscoveryConfig& rhs) {
    if (this != &rhs) {
        this->bcastAddress = rhs.bcastAddress;
        this->bindAddress = rhs.bindAddress;
        this->responseAddress = rhs.responseAddress;
    }

    return *this;
}


/*
 * vislib::net::cluster::DiscoveryService::DiscoveryConfig::operator ==
 */
bool vislib::net::cluster::DiscoveryService::DiscoveryConfig::operator ==(
        const DiscoveryConfig& rhs) const {
    return ((this->bcastAddress == rhs.bcastAddress)
        && (this->bindAddress == rhs.bindAddress)
        && (this->responseAddress == rhs.responseAddress));
}

// End class DiscoveryConfig
////////////////////////////////////////////////////////////////////////////////


/*
 * vislib::net::cluster::DiscoveryService::DEFAULT_PORT 
 */
const USHORT vislib::net::cluster::DiscoveryService::DEFAULT_PORT = 28181;


/*
 * vislib::net::cluster::DiscoveryService::DEFAULT_REQUEST_INTERVAL
 */
const UINT vislib::net::cluster::DiscoveryService::DEFAULT_REQUEST_INTERVAL
    = 10 * 1000;


/*
 * vislib::net::cluster::DiscoveryService::DEFAULT_RESPONSE_CHANCES
 */
const UINT vislib::net::cluster::DiscoveryService::DEFAULT_RESPONSE_CHANCES = 1;


/*
 * vislib::net::cluster::DiscoveryService::FLAG_OBSERVE_ONLY
 */
const UINT32 vislib::net::cluster::DiscoveryService::FLAG_OBSERVE_ONLY = 0x1;


/*
 * vislib::net::cluster::DiscoveryService::FLAG_SHARE_SOCKETS
 */
const UINT32 vislib::net::cluster::DiscoveryService::FLAG_SHARE_SOCKETS = 0x2;


/*
 * vislib::net::cluster::DiscoveryService::MSG_TYPE_USER
 */
const UINT32 vislib::net::cluster::DiscoveryService::MSG_TYPE_USER = 16;


/*
 * vislib::net::cluster::DiscoveryService::DiscoveryService
 */
vislib::net::cluster::DiscoveryService::DiscoveryService(void) 
        : configs(0), peerNodes(0) {
}


/*
 * vislib::net::cluster::DiscoveryService::~DiscoveryService
 */
vislib::net::cluster::DiscoveryService::~DiscoveryService(void) {
    try {
        this->Stop();
    } catch (...) {
        VLTRACE(Trace::LEVEL_VL_WARN, "Stopping the discovery threads crashed "
            "in the dtor. This should never happen.\n");
    }
}


/*
 * vislib::net::cluster::DiscoveryService::AddListener
 */
void vislib::net::cluster::DiscoveryService::AddListener(
        DiscoveryListener *listener) {
    ASSERT(listener != NULL);

    this->listenersCritSect.Lock();
    if ((listener != NULL) && !this->listeners.Contains(listener)) {
        this->listeners.Append(listener);
    }
    this->listenersCritSect.Unlock();
}


/*
 * vislib::net::cluster::DiscoveryService::GetDiscoveryAddress4
 */
vislib::net::IPAddress 
vislib::net::cluster::DiscoveryService::GetDiscoveryAddress4(
        const PeerHandle& hPeer) const {
    IPAddress retval = IPAddress::NONE;
    
    this->peerNodesCritSect.Lock();

    if ((hPeer == NULL) || !hPeer->isValid()) {
        this->peerNodesCritSect.Unlock();
        throw IllegalParamException("hPeer", __FILE__, __LINE__);
    } else {
        retval = hPeer->getDiscoveryAddress4();
        this->peerNodesCritSect.Unlock();
    }

    return retval;
}


/*
 * vislib::net::cluster::DiscoveryService::GetDiscoveryAddress6
 */
vislib::net::IPAddress6 
vislib::net::cluster::DiscoveryService::GetDiscoveryAddress6(
        const PeerHandle& hPeer) const {
    IPAddress6 retval;
    
    this->peerNodesCritSect.Lock();

    if ((hPeer == NULL) || !hPeer->isValid()) {
        this->peerNodesCritSect.Unlock();
        throw IllegalParamException("hPeer", __FILE__, __LINE__);
    } else {
        retval = hPeer->getDiscoveryAddress6();
        this->peerNodesCritSect.Unlock();
    }

    return retval;
}


/*
 * vislib::net::cluster::DiscoveryService::IsRunning
 */
bool vislib::net::cluster::DiscoveryService::IsRunning(void) const {
    SIZE_T cntConfigs = this->configs.Count();
    // TODO: reimplement this with a state enumeration
    
    if (!this->senderThread.IsRunning()) {
        return false;
    }

    for (SIZE_T i = 0; i < cntConfigs; i++) {
        if (!this->configs[i].GetRecvThread().IsRunning()) {
            return false;
        }
    }

    return true;
}


/*
 * vislib::net::cluster::DiscoveryService::IsStopped
 */
bool vislib::net::cluster::DiscoveryService::IsStopped(void) const {
    // TODO: reimplement this with a state enumeration
    return !this->IsRunning();
}


/*
 * vislib::net::cluster::DiscoveryService::RemoveListener
 */
void vislib::net::cluster::DiscoveryService::RemoveListener(
        DiscoveryListener *listener) {
    ASSERT(listener != NULL);

    this->listenersCritSect.Lock();
    this->listeners.RemoveAll(listener);
    this->listenersCritSect.Unlock();
}


/*
 * vislib::net::cluster::DiscoveryService::SendUserMessage
 */
UINT vislib::net::cluster::DiscoveryService::SendUserMessage(
        const UINT32 msgType, const void *msgBody, const SIZE_T msgSize) {
    Message msg;                // The datagram we are going to send.
    UINT retval = 0;            // Number of failed communication trials.

    this->prepareUserMessage(msg, msgType, msgBody, msgSize);

    /* Send the message to all registered clients. */
    this->peerNodesCritSect.Lock();
    for (SIZE_T i = 0; i < this->peerNodes.Count(); i++) {
        try {
            this->peerNodes[i]->discoverySource->SendMessageTo(msg,
                this->peerNodes[i]->discoveryAddress);
        } catch (SocketException e) {
            VLTRACE(Trace::LEVEL_VL_WARN, "DiscoveryService could not send "
                "user message (\"%s\").\n", e.GetMsgA());
            retval++;
        }
    }
    this->peerNodesCritSect.Unlock();

    return retval;
}


/*
 * vislib::net::cluster::DiscoveryService::SendUserMessage
 */
UINT vislib::net::cluster::DiscoveryService::SendUserMessage(
        const PeerHandle& hPeer, const UINT32 msgType, const void *msgBody, 
        const SIZE_T msgSize) {
    Message msg;                // The datagram we are going to send.
    UINT retval = 0;            // Retval, 0 in case of success.

    this->prepareUserMessage(msg, msgType, msgBody, msgSize);
    
    this->peerNodesCritSect.Lock();

    if ((hPeer == NULL) || !hPeer->isValid()) {
        this->peerNodesCritSect.Unlock();
        throw IllegalParamException("hPeer", __FILE__, __LINE__);
    }

    try {
        hPeer->discoverySource->SendMessageTo(msg, hPeer->discoveryAddress);
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_WARN, "DiscoveryService could not send "
            "user message (\"%s\").\n", e.GetMsgA());
        retval++;
    }
    this->peerNodesCritSect.Unlock();

    return retval;
}


/*
 * vislib::net::cluster::DiscoveryService::Start
 */
void vislib::net::cluster::DiscoveryService::Start(const char *name, 
        const DiscoveryConfig *configs, const SIZE_T cntConfigs,
        const UINT32 flags,
        const UINT requestInterval,
        const UINT cntResponseChances) {
    // TODO: Check for restart

    this->name = name;
    this->name.Truncate(MAX_NAME_LEN);

    this->configs.AssertCapacity(cntConfigs);
    for (SIZE_T i = 0; i < cntConfigs; i++) {
        this->configs.Add(DiscoveryConfigEx(configs[i], this));
    }

    this->flags = flags;
    this->requestInterval = requestInterval;
    this->cntResponseChances = cntResponseChances;

    /* Start the threads. */
    this->senderThread.Start(this);
    ASSERT(cntConfigs == this->configs.Count());
    for (SIZE_T i = 0; i < cntConfigs; i++) {
        this->configs[i].GetRecvThread().Start(&(this->configs[i]));
    }
}


/*
 * vislib::net::cluster::DiscoveryService::Stop
 */
bool vislib::net::cluster::DiscoveryService::Stop(const bool noWait) {
    SIZE_T cntConfigs = this->configs.Count();
    bool retval = true;
    
    // Note: Receiver must be stopped first, in order to prevent shutdown
    // deadlock when waiting for the last message.    
    for (SIZE_T i = 0; i < cntConfigs; i++) {
        try {
            this->configs[i].GetRecvThread().Terminate(false);
        } catch (sys::SystemException e) {
            VLTRACE(Trace::LEVEL_VL_ERROR, "Stopping a discovery receiver "
                "thread failed. The error code is %d (\"%s\").\n", 
                e.GetErrorCode(), e.GetMsgA());
            retval = false;
        }
    }
    
    try {
        this->senderThread.TryTerminate(false);
    } catch (sys::SystemException e) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Stopping the discovery sender thread "
            "failed. The error code is %d (\"%s\").\n", 
            e.GetErrorCode(), e.GetMsgA());
        retval = false;
    }

    return retval;
}


/*
 * vislib::net::cluster::DiscoveryService::operator []
 */
vislib::net::IPEndPoint vislib::net::cluster::DiscoveryService::operator [](
        const PeerHandle& hPeer) const {
    IPEndPoint retval;
    
    this->peerNodesCritSect.Lock();

    if ((hPeer == NULL) || !hPeer->isValid()) {
        this->peerNodesCritSect.Unlock();
        throw IllegalParamException("hPeer", __FILE__, __LINE__);
    } else {
        retval = hPeer->responseAddress;
        this->peerNodesCritSect.Unlock();
    }

    return retval;
}


////////////////////////////////////////////////////////////////////////////////
// Begin class Receiver

/*
 * vislib::net::cluster::DiscoveryService::Receiver::Receiver
 */
vislib::net::cluster::DiscoveryService::Receiver::Receiver(void) 
        : isRunning(1) {
}


/*
 * vislib::net::cluster::DiscoveryService::Receiver::~Receiver
 */
vislib::net::cluster::DiscoveryService::Receiver::~Receiver(void) {
}


/*
 * vislib::net::cluster::DiscoveryService::Receiver::Run
 */
DWORD vislib::net::cluster::DiscoveryService::Receiver::Run(void *dcfg) {
    IPEndPoint peerAddr;            // Receives address of communication peer.
    PeerNode peerNode;              // The peer node to register in our list.
    Message msg;                    // Receives the request messages.
    DiscoveryConfigEx *config = static_cast<DiscoveryConfigEx *>(dcfg);

    ASSERT(config != NULL);
    ASSERT(!this->socket.IsValid());

    // Assert expected message memory layout.
    ASSERT(sizeof(msg) == MAX_USER_DATA + 2 * sizeof(UINT32));
    ASSERT(reinterpret_cast<BYTE *>(&(msg.SenderBody)) 
       == reinterpret_cast<BYTE *>(&msg) + 2 * sizeof(UINT32));

    /* 
     * Prepare a datagram socket listening for requests on the specified 
     * adapter and port. 
     */
    VLTRACE(Trace::LEVEL_VL_VERBOSE, "The discovery receiver thread is "
        "preparing its socket on %s ...\n", 
        config->GetBindAddress().ToStringA().PeekBuffer());
    try {
        Socket::Startup();
        this->socket.Create(config->GetProtocolFamily(), Socket::TYPE_DGRAM, 
            Socket::PROTOCOL_UDP);
        this->socket.SetBroadcast(true);
        this->socket.SetReuseAddr(true);
        this->socket.SetExclusiveAddrUse(
            !config->GetDiscoveryService().IsShareSockets());
        this->socket.Bind(config->GetBindAddress());
        //socket.SetLinger(false, 0);     // Force hard close.
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Discovery receiver thread could not "
            "create its socket and bind it to the requested address. The "
            "error code is %d (\"%s\").\n", e.GetErrorCode(), e.GetMsgA());
        return e.GetErrorCode();
    }

    /* Register myself as known node first (observer does not know itself). */
    if (!config->GetDiscoveryService().IsObserver()) {
        IPEndPoint discoveryAddr(config->GetResponseAddress());
        discoveryAddr.SetPort(config->GetBindAddress().GetPort());  // TODO: hugly
        VLTRACE(Trace::LEVEL_VL_VERBOSE, "Add myself (%s) as known node.\n",
            config->GetResponseAddress().ToStringA().PeekBuffer());
        config->GetDiscoveryService().addPeerNode(discoveryAddr,
            config, config->GetResponseAddress());
    }

    VLTRACE(Trace::LEVEL_VL_INFO, "The discovery receiver thread is "
        "starting ...\n");

    while (this->isRunning != 0) {
        try {

            /* Wait for next message. */
            this->socket.Receive(peerAddr, &msg, sizeof(Message));

            if (msg.MagicNumber != MAGIC_NUMBER) {
                VLTRACE(Trace::LEVEL_VL_ERROR, "Discovery service received "
                    "invalid magic number %d from %s.\n", msg.MagicNumber,
                    peerAddr.ToStringA().PeekBuffer());
                continue;
            }

            if (msg.MsgType >= MSG_TYPE_USER) {
                /* 
                 * Received user message. This is independent of the cluster 
                 * name as the messages are directed directly to the nodes
                 * rather than to the broadcast address.
                 */
                config->GetDiscoveryService().fireUserMessage(peerAddr, 
                    msg.MsgType, msg.UserData);
                continue;
            }

            if (!config->GetDiscoveryService().GetName().Equals(
                    msg.SenderBody.Name)) {
                VLTRACE(Trace::LEVEL_VL_ERROR, "Discovery service received "
                    "message for different cluster \"%s\" from %s.\n", 
                    msg.SenderBody.Name,
                    peerAddr.ToStringA().PeekBuffer());
                continue;
            }


            switch (msg.MsgType) {

                case MSG_TYPE_IAMALIVE:
                    VLTRACE(Trace::LEVEL_VL_INFO, "Discovery service received "
                        "MSG_TYPE_IAMALIVE from %s.\n", 
                        peerAddr.ToStringA().PeekBuffer());
                    
                    /* Add peer to local list, if not yet known. */
                    config->GetDiscoveryService().addPeerNode(peerAddr, 
                        config,
                        IPEndPoint(msg.SenderBody.ResponseAddress));
                    break;

                case MSG_TYPE_IAMHERE:
                    /* 
                     * Get an initial discovery request. This triggers an 
                     * immediate alive message, but without adding the sender to
                     * the cluster.
                     *
                     * Note: Nodes sending the MSG_TYPE_IAMHERE message must
                     * be added to the list of known nodes, because nodes in 
                     * observer mode also send MSG_TYPE_IAMHERE to request an
                     * immediate response of all running nodes.
                     */
                    VLTRACE(Trace::LEVEL_VL_INFO, "Discovery service received "
                        "MSG_TYPE_IAMHERE from %s.\n", 
                        peerAddr.ToStringA().PeekBuffer());

                    if (!config->GetDiscoveryService().IsObserver()) {
                        /* Observers must not send alive messages. */
                        peerAddr.SetPort(config->GetBindAddress().GetPort());
                        ASSERT(msg.MagicNumber == MAGIC_NUMBER);
                        msg.MsgType = MSG_TYPE_IAMALIVE;
                        msg.SenderBody.ResponseAddress 
                            = config->GetResponseAddress();
                        this->socket.Send(peerAddr, &msg, sizeof(Message));
                        VLTRACE(Trace::LEVEL_VL_INFO, "Discovery service sent "
                            "immediate MSG_TYPE_IAMALIVE answer to %s.\n",
                            peerAddr.ToStringA().PeekBuffer());
                    }
                    break;

                case MSG_TYPE_SAYONARA:
                    /* Got an explicit disconnect. */
                    VLTRACE(Trace::LEVEL_VL_INFO, "Discovery service received "
                        "MSG_TYPE_SAYONARA from %s.\n",
                        peerAddr.ToStringA().PeekBuffer());

                    config->GetDiscoveryService().removePeerNode(
                        IPEndPoint(msg.SenderBody.ResponseAddress));
                    break;
            }

        } catch (SocketException e) {
            VLTRACE(Trace::LEVEL_VL_WARN, "A socket error occurred in the "
                "discovery receiver thread. The error code is %d (\"%s\").\n",
                e.GetErrorCode(), e.GetMsgA());
        } catch (...) {
            VLTRACE(Trace::LEVEL_VL_ERROR, "The discovery receiver caught an "
                "unexpected exception and will exit.\n");
            return -1;
        }

    } /* end while (this->isRunning) */

    try {
        Socket::Cleanup();
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Socket cleanup failed in the discovery "
            "receiver thread. The error code is %d (\"%s\").\n", 
            e.GetErrorCode(), e.GetMsgA());
        return e.GetErrorCode();
    }

    return 0;
}


/*
 * vislib::net::cluster::DiscoveryService::Receiver::Terminate
 */
bool vislib::net::cluster::DiscoveryService::Receiver::Terminate(void) {
    vislib::sys::Interlocked::Exchange(&this->isRunning, 0);
    this->socket.Close();   // TODO: Should perhaps be protected by crit sect?
    return true;
}

// End class Receiver
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Begin class DiscoveryConfigEx

/*
 * vislib::net::cluster::DiscoveryService::DiscoveryConfigEx::DiscoveryConfigEx
 */
vislib::net::cluster::DiscoveryService::DiscoveryConfigEx::DiscoveryConfigEx(
        void) : Super(IPEndPoint(), IPAddress::ANY, IPAddress::ANY, 
        DEFAULT_PORT), cds(NULL) {
}


/*
 * vislib::net::cluster::DiscoveryService::DiscoveryConfigEx::DiscoveryConfigEx
 */
vislib::net::cluster::DiscoveryService::DiscoveryConfigEx::DiscoveryConfigEx(
        const DiscoveryConfig& config, DiscoveryService *cds) 
        : Super(config), cds(cds) {
    ASSERT(cds != NULL);
}


/*
 * vislib::net::cluster::DiscoveryService::DiscoveryConfigEx::DiscoveryConfigEx
 */
vislib::net::cluster::DiscoveryService::DiscoveryConfigEx::DiscoveryConfigEx(
        const DiscoveryConfigEx& rhs) 
        : Super(rhs), cds(rhs.cds) {
}


/*
 * vislib::net::cluster::DiscoveryService::DiscoveryConfigEx::~DiscoveryConfigEx
 */
vislib::net::cluster::DiscoveryService::DiscoveryConfigEx::~DiscoveryConfigEx(
        void) {
    try {
        this->socketSend.Close();
    } catch (...) {
    }
    try {
        this->socketUserMsg.Close();
    } catch (...) {
    }
    try {
        // TODO: Check this 
        this->recvThread.Terminate(true);
    } catch (...) {
        VLTRACE(Trace::LEVEL_VL_WARN, "Error while forcefully terminating "
            "receiver thread.\n");
    }
}


/*
 * ...::net::cluster::DiscoveryService::DiscoveryConfigEx::SendCustomisedMessage
 */
void 
vislib::net::cluster::DiscoveryService::DiscoveryConfigEx::SendCustomisedMessage(
        Message& message) {
    switch (message.MsgType) {
        case MSG_TYPE_IAMHERE:
            /* falls through. */
        case MSG_TYPE_IAMALIVE:
            /* falls through. */
        case MSG_TYPE_SAYONARA:
            message.SenderBody.ResponseAddress = this->responseAddress;
            break;

        default:
            break;
    }

    this->SendMessageTo(message, this->bcastAddress);
}


/*
 * vislib::net::cluster::DiscoveryService::DiscoveryConfigEx::SendMessageTo
 */
void 
vislib::net::cluster::DiscoveryService::DiscoveryConfigEx::SendMessageTo(
        Message& message, const IPEndPoint& to) {
    Socket *socket = NULL;

    if (message.MsgType >= MSG_TYPE_USER) {
        if (!this->socketSend.IsValid()) {
            // Create a bind address for the sender socket from the receiver 
            // address.
            IPEndPoint bindAddr(this->bindAddress, 0);

            try {
                this->socketSend.Create(this->GetProtocolFamily(), 
                    Socket::TYPE_DGRAM, Socket::PROTOCOL_UDP);
                this->socketSend.SetBroadcast(true);
                //this->socketSend.SetLinger(false, 0);     // Force hard close.
                //this->socketSend.Bind(bindAddr);
            } catch (SocketException e) {
                VLTRACE(Trace::LEVEL_VL_ERROR, "The socket for the discovery "
                    "sender could not be created. The error code is %d "
                    "(\"%s\").\n", e.GetErrorCode(), e.GetMsgA());
                throw;
            }
        }

        socket = &this->socketSend;

    } else {
        if (!this->socketUserMsg.IsValid()) {
            try {
                this->socketUserMsg.Create(this->GetProtocolFamily(), 
                    Socket::TYPE_DGRAM, Socket::PROTOCOL_UDP);
            } catch (SocketException e) {
                VLTRACE(Trace::LEVEL_VL_ERROR, "The socket for sending user "
                    "messages could not be created. The error code is %d "
                    "(\"%s\").\n", e.GetErrorCode(), e.GetMsgA());
                throw;
            } 
        }

        socket = &this->socketUserMsg;
    }

    socket->Send(to, &message, sizeof(Message));  
}


/*
 * vislib::net::cluster::DiscoveryService::DiscoveryConfigEx::operator =
 */
vislib::net::cluster::DiscoveryService::DiscoveryConfigEx& 
vislib::net::cluster::DiscoveryService::DiscoveryConfigEx::operator =(
        const DiscoveryConfigEx& rhs) {
    if (this != &rhs) {
        Super::operator =(rhs);
        this->cds = rhs.cds;
        this->socketSend = rhs.socketSend;
    }

    return *this;
}


/*
 * vislib::net::cluster::DiscoveryService::DiscoveryConfigEx::operator ==
 */
bool vislib::net::cluster::DiscoveryService::DiscoveryConfigEx::operator ==(
        const DiscoveryConfigEx& rhs) const {
    if (Super::operator ==(rhs)) {
        return ((this->cds == rhs.cds)
            && (this->socketSend == rhs.socketSend));

    } else {
        return false;
    }
}

// End class DiscoveryConfigEx
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Begin class Sender

/*
 * vislib::net::cluster::DiscoveryService::Sender::Sender
 */
vislib::net::cluster::DiscoveryService::Sender::Sender(void) : isRunning(1) {
}


/*
 * vislib::net::cluster::DiscoveryService::Sender::~Sender
 */
vislib::net::cluster::DiscoveryService::Sender::~Sender(void) {
}


/*
 * vislib::net::cluster::DiscoveryService::Sender::Run
 */
DWORD vislib::net::cluster::DiscoveryService::Sender::Run(void *cds) {
    ASSERT(cds != NULL);

    Message request;        // The UDP datagram we send.
    DiscoveryService *ds    // The discovery service we work for.
        = static_cast<DiscoveryService *>(cds);
    SIZE_T cntConfigs = ds->configs.Count();    // # of configs. to serve.

    // Assert expected memory layout of messages.
    ASSERT(sizeof(request) == MAX_USER_DATA + 2 * sizeof(UINT32));
    ASSERT(reinterpret_cast<BYTE *>(&(request.SenderBody)) 
       == reinterpret_cast<BYTE *>(&request) + 2 * sizeof(UINT32));

    VLTRACE(Trace::LEVEL_VL_INFO, "The discovery sender thread is "
        "starting ...\n");

    /* Prepare the sockets. */
    try {
        Socket::Startup();
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Discovery sender thread could not "
            "create initialise socket sub-system. The error code is %d "
            "(\"%s\").\n", e.GetErrorCode(), e.GetMsgA());
        return e.GetErrorCode();
    }

    /* Prepare the constant parameters of our requests. */
    request.MagicNumber = MAGIC_NUMBER;
#if (_MSC_VER >= 1400)
    ::strncpy_s(request.SenderBody.Name, MAX_NAME_LEN, ds->name.PeekBuffer(),
        MAX_NAME_LEN);
#else /* (_MSC_VER >= 1400) */
    ::strncpy(request.SenderBody.Name, ds->Name.PeekBuffer(), MAX_NAME_LEN);
#endif /* (_MSC_VER >= 1400) */


    /* Send the initial "immediate alive request" message(s). */
    request.MsgType = MSG_TYPE_IAMHERE;
    for (SIZE_T i = 0; i < cntConfigs; i++) {
        DiscoveryConfigEx& config = ds->configs[i];

        if (this->isRunning == 0) {
            break;
        }
        
        try {
            config.SendCustomisedMessage(request);
            VLTRACE(Trace::LEVEL_VL_INFO, "Discovery service sent "
                "MSG_TYPE_IAMHERE to %s.\n", 
                config.GetBcastAddress().ToStringA().PeekBuffer());
        } catch (SocketException e) {
            VLTRACE(Trace::LEVEL_VL_ERROR, "A socket error occurred in the "
                "discovery sender thread. The error code is %d (\"%s\").\n",
                e.GetErrorCode(), e.GetMsgA());
        }
    } /* end for (SIZE_T i = 0; i < cntConfigs; i++) */


    /*
     * If the discovery service is configured not to be member of the 
     * cluster, but to only search other members, the sender thread can 
     * leave after the first request sent above.
     * Otherwise, the thread should wait for the normal request interval
     * time before really starting.
     */
    if (ds->IsObserver()) {
        VLTRACE(Trace::LEVEL_VL_INFO, "Discovery service is leaving the sender "
            "thread as it is in observer mode.\n");
        return 0;
    } else {
        sys::Thread::Sleep(ds->requestInterval);
    }


    /* Change our request for alive broadcasting. */
    request.MsgType = MSG_TYPE_IAMALIVE;
    while (this->isRunning != 0) {
        try {
            ds->prepareRequest();
        } catch (...) {
            VLTRACE(Trace::LEVEL_VL_ERROR, "The discovery sender thread caught "
                "an unexpected exception while preparing the request.\n");
            return -1;
        }

        for (SIZE_T i = 0; i < cntConfigs; i++) {
            DiscoveryConfigEx& config = ds->configs[i];

            if (this->isRunning == 0) {
                break;
            }

            try {
                config.SendCustomisedMessage(request);
                VLTRACE(Trace::LEVEL_VL_INFO, "Discovery service sent "
                    "MSG_TYPE_IAMALIVE to %s.\n", 
                    config.GetBcastAddress().ToStringA().PeekBuffer());
            } catch (SocketException e) {
                VLTRACE(Trace::LEVEL_VL_ERROR, "A socket error occurred in the "
                    "discovery sender thread. The error code is %d (\"%s\").\n",
                    e.GetErrorCode(), e.GetMsgA());
            }
        } /* end for (SIZE_T i = 0; i < cntConfigs; i++) */ 

        try {
            if (this->isRunning != 0) {
                sys::Thread::Sleep(ds->requestInterval);
            }
        } catch (sys::SystemException e) {
            VLTRACE(Trace::LEVEL_VL_WARN, "An error occurred in the "
                "discovery sender thread while waiting. The error code is %d "
                "(\"%s\").\n", e.GetErrorCode(), e.GetMsgA());
        }
    } /* end while (this->isRunning != 0) */

    /* Now inform all other nodes, that we are out. */
    request.MsgType = MSG_TYPE_SAYONARA;
    for (SIZE_T i = 0; i < cntConfigs; i++) {
        DiscoveryConfigEx& config = ds->configs[i];

        try {
            config.SendCustomisedMessage(request);
            VLTRACE(Trace::LEVEL_VL_INFO, "Discovery service sent "
                "MSG_TYPE_SAYONARA to %s.\n", 
                config.GetBcastAddress().ToStringA().PeekBuffer());
        } catch (SocketException e) {
            VLTRACE(Trace::LEVEL_VL_ERROR, "A socket error occurred in the "
                "discovery sender thread. The error code is %d (\"%s\").\n",
                e.GetErrorCode(), e.GetMsgA());
        }
    } /* end for (SIZE_T i = 0; i < cntConfigs; i++) */

    /* Clean up. */
    try {
        Socket::Cleanup();
    } catch (SocketException e) {
        VLTRACE(Trace::LEVEL_VL_ERROR, "Socket cleanup failed in the discovery "
            "request thread. The error code is %d (\"%s\").\n", 
            e.GetErrorCode(), e.GetMsgA());
        return e.GetErrorCode();
    }

    return 0;
}


/*
 * vislib::net::cluster::DiscoveryService::Sender::Terminate
 */
bool vislib::net::cluster::DiscoveryService::Sender::Terminate(void) {
    vislib::sys::Interlocked::Exchange(&this->isRunning, 0);
    return true;
}


// End class Sender
////////////////////////////////////////////////////////////////////////////////



/*
 * vislib::net::cluster::DiscoveryService::addPeerNode
 */
void vislib::net::cluster::DiscoveryService::addPeerNode(
        const IPEndPoint& discoveryAddr, 
        DiscoveryConfigEx *discoverySource,
        const IPEndPoint& responseAddr) {
    INT_PTR idx = 0;            // Index of possible duplicate.
    PeerHandle hPeer = NULL;    // Old or new peer node handle.

    ASSERT(discoverySource != NULL);

    this->peerNodesCritSect.Lock();
    
    if ((idx = this->peerFromResponseAddress(responseAddr)) >= 0) {
        /* Already known, reset disconnect chance. */
        VLTRACE(Trace::LEVEL_VL_VERBOSE, "Peer node %s is already known.\n", 
            responseAddr.ToStringA().PeekBuffer());
        hPeer = this->peerNodes[static_cast<SIZE_T>(idx)];
        hPeer->cntResponseChances = this->cntResponseChances;

    } else {
        /* Not known, so add it and fire event. */
        VLTRACE(Trace::LEVEL_VL_INFO, "I learned about node %s.\n",
            responseAddr.ToStringA().PeekBuffer());
        hPeer = new PeerNode(discoveryAddr, responseAddr, 
            this->cntResponseChances, discoverySource);
        this->peerNodes.Append(hPeer);

        this->listenersCritSect.Lock();
        ConstIterator<ListenerList::Iterator> it 
            = this->listeners.GetConstIterator();
        while (it.HasNext()) {
            it.Next()->OnNodeFound(*this, hPeer);
        }
        this->listenersCritSect.Unlock();
    }

    this->peerNodesCritSect.Unlock();
}


/*
 * vislib::net::cluster::DiscoveryService::fireUserMessage
 */
void vislib::net::cluster::DiscoveryService::fireUserMessage(
        const IPEndPoint& sender, const UINT32 msgType, 
        const BYTE *msgBody) const {
    INT_PTR idx = 0;        // Index of sender PeerNode.

    this->peerNodesCritSect.Lock();
    
    if ((idx = this->peerFromDiscoveryAddr(sender)) >= 0) {
        this->listenersCritSect.Lock();

        ConstIterator<ListenerList::Iterator> it 
            = this->listeners.GetConstIterator();
        while (it.HasNext()) {
            it.Next()->OnUserMessage(*this, 
                this->peerNodes[static_cast<SIZE_T>(idx)], 
                msgType,
                msgBody);
        }

        this->listenersCritSect.Unlock();
    }
    
    this->peerNodesCritSect.Lock();
}


/*
 * vislib::net::cluster::DiscoveryService::peerFromResponseAddress
 */
INT_PTR vislib::net::cluster::DiscoveryService::peerFromResponseAddress(
        const IPEndPoint& addr) const {
    // TODO: Think of faster solution.  
    INT_PTR retval = -1;

    for (SIZE_T i = 0; i < this->peerNodes.Count(); i++) {
        if (this->peerNodes[i]->responseAddress == addr) {
            retval = i;
            break;
        }
    }

    return retval;
}


/*
 * vislib::net::cluster::DiscoveryService::peerFromDiscoveryAddr
 */
INT_PTR vislib::net::cluster::DiscoveryService::peerFromDiscoveryAddr(
        const IPEndPoint& addr) const {
    // TODO: Think of faster solution.  
    INT_PTR retval = -1;

    for (SIZE_T i = 0; i < this->peerNodes.Count(); i++) {
        if (this->peerNodes[i]->discoveryAddress == addr) {
            retval = i;
            break;
        }
    }

    return retval;
}


/*
 * vislib::net::cluster::DiscoveryService::prepareRequest
 */
void vislib::net::cluster::DiscoveryService::prepareRequest(void) {
    this->peerNodesCritSect.Lock();

    for (SIZE_T i = 0; i < this->peerNodes.Count(); i++) {
        if (this->peerNodes[i]->decrementResponseChances()) {
            
            /* Fire event. */
            ConstIterator<ListenerList::Iterator> it 
                = this->listeners.GetConstIterator();
            while (it.HasNext()) {
                it.Next()->OnNodeLost(*this, this->peerNodes[i],
                    DiscoveryListener::LOST_IMLICITLY);
            }

            /*
             * Make the handle invalid (there might be other smart 
             * pointers == handles out there). 
             */
            this->peerNodes[i]->invalidate();

            /* Remove peer from list. */
            this->peerNodes.Erase(i);
            i--;
        }
    }

    this->peerNodesCritSect.Unlock();
}


/*
 * vislib::net::cluster::DiscoveryService::prepareUserMessage
 */
void vislib::net::cluster::DiscoveryService::prepareUserMessage(
        Message& outMsg, const UINT32 msgType, const void *msgBody, 
        const SIZE_T msgSize) {

    /* Check user parameters. */
    if (msgType < MSG_TYPE_USER) {
        throw IllegalParamException("msgType", __FILE__, __LINE__);
    }
    if (msgBody == NULL) {
        throw IllegalParamException("msgBody", __FILE__, __LINE__);
    }
    if (msgSize > MAX_USER_DATA) {
        throw IllegalParamException("msgSize", __FILE__, __LINE__);
    }

    // Assert some stuff.
    ASSERT(sizeof(outMsg) == MAX_USER_DATA + 2 * sizeof(UINT32));
    ASSERT(reinterpret_cast<BYTE *>(&(outMsg.UserData)) 
       == reinterpret_cast<BYTE *>(&outMsg) + 2 * sizeof(UINT32));
    ASSERT(msgType >= MSG_TYPE_USER);
    ASSERT(msgBody != NULL);
    ASSERT(msgSize <= MAX_USER_DATA);

    /* Prepare the message. */
    outMsg.MagicNumber = MAGIC_NUMBER;
    outMsg.MsgType = msgType;
    ::ZeroMemory(outMsg.UserData, MAX_USER_DATA);
    ::memcpy(outMsg.UserData, msgBody, msgSize);
}


/*
 * vislib::net::cluster::DiscoveryService::removePeerNode
 */
void vislib::net::cluster::DiscoveryService::removePeerNode(
        const IPEndPoint& address) {
    INT_PTR idx = 0;
    
    this->listenersCritSect.Lock();

    if ((idx = this->peerFromResponseAddress(address)) >= 0) {
        PeerHandle hPeer = this->peerNodes[static_cast<SIZE_T>(idx)];

        /* Fire event. */
        ConstIterator<ListenerList::Iterator> it 
            = this->listeners.GetConstIterator();
        while (it.HasNext()) {
            it.Next()->OnNodeLost(*this, hPeer, 
                DiscoveryListener::LOST_EXPLICITLY);
        }

        /* Invalidate handle. */
        hPeer->invalidate();

        /* Remove peer from list. */
        this->peerNodes.Erase(idx);
    }

    this->listenersCritSect.Unlock();
}


/*
 * vislib::net::cluster::DiscoveryService::MAGIC_NUMBER
 */
const UINT32 vislib::net::cluster::DiscoveryService::MAGIC_NUMBER 
    = (static_cast<UINT32>('v') << 0) | (static_cast<UINT32>('c') << 8)
    | (static_cast<UINT32>('d') << 16) | (static_cast<UINT32>('s') << 24);


#ifndef _WIN32
/*
 * vislib::net::cluster::DiscoveryService::MSG_TYPE_IAMALIVE
 */
const UINT32 vislib::net::cluster::DiscoveryService::MSG_TYPE_IAMALIVE;


/*
 * vislib::net::cluster::DiscoveryService::MSG_TYPE_IAMHERE
 */
const UINT32 vislib::net::cluster::DiscoveryService::MSG_TYPE_IAMHERE;


/*
 * vislib::net::cluster::DiscoveryService::MSG_TYPE_SAYONARA
 */
const UINT32 vislib::net::cluster::DiscoveryService::MSG_TYPE_SAYONARA;
#endif  /* !_WIN32 */
