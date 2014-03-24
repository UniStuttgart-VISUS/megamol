/*
 * AbstractServerNode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/AbstractServerNode.h"

#include "the/assert.h"
#include "the/argument_exception.h"
#include "the/index_out_of_range_exception.h"
#include "the/no_such_element_exception.h"
#include "vislib/SocketException.h"
#include "the/system/system_exception.h"
#include "the/trace.h"
#include "vislib/unreferenced.h"
#include "the/not_supported_exception.h"

#include "messagereceiver.h"


/*
 * vislib::net::cluster::AbstractServerNode::~AbstractServerNode
 */
vislib::net::cluster::AbstractServerNode::~AbstractServerNode(void) {
    try {
        this->server.Terminate(false);

        while (this->countPeers() > 0) {
            this->disconnectPeer(0);
        }
    } catch (the::exception e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Exception while releasing "
            "ServerNodeAdapter: %s\n", e.what());
    } catch (...) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Unexpected exception whie releasing "
            "ServerNodeAdapter.\n");
    }
}

/*
 * vislib::net::cluster::AbstractServerNode::GetBindAddress
 */
const vislib::net::IPEndPoint& 
vislib::net::cluster::AbstractServerNode::GetBindAddress(void) const {
    return this->bindAddress;
}


/*
 * vislib::net::cluster::AbstractServerNode::Initialise
 */
void vislib::net::cluster::AbstractServerNode::Initialise(
        sys::CmdLineProviderA& inOutCmdLine) {
    // Do not call superclass method as it is not initalised.
    this->initialise(inOutCmdLine);
}


/*
 * vislib::net::cluster::AbstractServerNode::Initialise
 */
void vislib::net::cluster::AbstractServerNode::Initialise(
        sys::CmdLineProviderW& inOutCmdLine) {
    // Do not call superclass method as it is not initalised.
    this->initialise(inOutCmdLine);
}


/*
 * vislib::net::cluster::AbstractServerNode::OnNewConnection
 */
bool vislib::net::cluster::AbstractServerNode::OnNewConnection(Socket& socket,
        const IPEndPoint& addr) throw() {
    try {
        socket.SetNoDelay(true);

        PeerNode *peerNode = new PeerNode;
        peerNode->Socket = socket;
        peerNode->Receiver = new sys::Thread(ReceiveMessages);

        ReceiveMessagesCtx *rmc = AllocateRecvMsgCtx(this, &peerNode->Socket);
        try {
            THE_VERIFY(peerNode->Receiver->Start(static_cast<void *>(rmc)));
        } catch (the::exception e) {
            FreeRecvMsgCtx(rmc);
            throw e;
        }

        this->peersLock.Lock();
        this->peers.Add(peerNode);
        this->peersLock.Unlock();

        this->onPeerConnected(addr);

        return true;
    } catch (the::exception e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Could not accept peer node %s in "
            "ServerNodeAdapter because of an exception: %s\n", 
            addr.ToStringA().PeekBuffer(), e.what());
    } catch (...) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Could not accept peer node %s in "
            "ServerNodeAdapter because of an unexpected exception.\n", 
            addr.ToStringA().PeekBuffer());
    }
    /* Exception was caught if here. */

    return false;
}


/*
 * vislib::net::cluster::AbstractServerNode::OnServerStopped
 */
void vislib::net::cluster::AbstractServerNode::OnServerStopped(void) throw() {
}


/*
 * vislib::net::cluster::AbstractServerNode::Run
 */
unsigned int vislib::net::cluster::AbstractServerNode::Run(void) {
    bool isStarted = this->server.Start(&this->bindAddress);
#ifndef _WIN32
    isStarted = isStarted;
#endif /* !_WIN32 */
    // TODO: generate error here for restart?
    return 0;
}


/*
 * vislib::net::cluster::AbstractServerNode::SetBindAddress
 */
void vislib::net::cluster::AbstractServerNode::SetBindAddress(
        const IPEndPoint& bindAddress) {
    this->bindAddress = bindAddress;
}


/*
 * vislib::net::cluster::AbstractServerNode::SetBindAddress
 */
void vislib::net::cluster::AbstractServerNode::SetBindAddress(
        const unsigned short port) {
    // TODO: IPv6
    this->bindAddress = IPEndPoint(IPAddress::ANY, port);
}


/*
 * vislib::net::cluster::AbstractServerNode::AbstractServerNode
 */
vislib::net::cluster::AbstractServerNode::AbstractServerNode(void) 
        : AbstractClusterNode(), TcpServer::Listener(),
        // TODO: IPv6
        bindAddress(IPAddress::ANY, DEFAULT_PORT) {
    this->server.AddListener(this);
}


/*
 * vislib::net::cluster::AbstractServerNode::AbstractServerNode
 */
vislib::net::cluster::AbstractServerNode::AbstractServerNode(
        const AbstractServerNode& rhs) 
        : AbstractClusterNode(rhs), TcpServer::Listener(rhs) {
    throw the::not_supported_exception("AbstractServerNode", __FILE__,
        __LINE__);
}


/*
 * vislib::net::cluster::AbstractServerNode::countPeers
 */
size_t vislib::net::cluster::AbstractServerNode::countPeers(void) const {
    size_t retval = 0;
    this->peersLock.Lock();
    retval = this->peers.Count();
    this->peersLock.Unlock();
    return retval;
}


/*
 * vislib::net::cluster::AbstractServerNode::disconnectPeer
 */
void vislib::net::cluster::AbstractServerNode::disconnectPeer(
        const size_t idx) {
    this->peersLock.Lock();

    // We assure the user that erasing non-existent peers will silently fail, 
    // so catch all exception that we expect to occur here.
    try {
        PeerNode& peerNode = *this->peers[idx];
        peerNode.Socket.Close();
        the::safe_delete(peerNode.Receiver);
        this->peers.Erase(idx);

    } catch (SocketException se) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "SocketException occurred when "
            "disconnecting node %u: %s\n", idx, se.what());
    } catch (the::system::system_exception sye) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "the::system::system_exception occurred when "
            "disconnecting node %u: %s\n", idx, sye.what());
    } catch (the::index_out_of_range_exception oore) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "index_out_of_range_exception occurred when "
            "disconnecting node %u: %s\n", idx, oore.what());
    }

    this->peersLock.Unlock();
}


/*
 * vislib::net::cluster::AbstractServerNode::forEachPeer
 */
size_t vislib::net::cluster::AbstractServerNode::forEachPeer(
        ForeachPeerFunc func, void *context) {
    PeerIdentifier peerId;
    size_t retval = 0;

    this->peersLock.Lock();
    for (size_t i = 0; i < this->peers.Count(); i++) {
        try {
            peerId = this->peers[i]->Socket.GetPeerEndPoint();
            bool isContinue = func(this, peerId, this->peers[i]->Socket, 
                context);
            retval++;

            if (!isContinue) {
                break;
            }
        } catch (the::exception& e) {
            VL_DBGONLY_REFERENCED_LOCAL_VARIABLE(e);
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "ForeachPeerFunc failed for node %u "
                "(%s) with an exception: %s\n", i, 
                peerId.ToStringA().PeekBuffer(), e.what());
            // TODO: second chance??????
            this->disconnectPeer(i);
            i--;
        } catch (...) {
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "ForeachPeerFunc failed for node %u "
                "(%s) with a non-VISlib exception.\n", i,
                peerId.ToStringA().PeekBuffer());
            // TODO: second chance??????
            this->disconnectPeer(i);
            i--;
        }
    }
    this->peersLock.Unlock();

    return retval;
}


/*
 * vislib::net::cluster::AbstractServerNode::forPeer
 */
bool vislib::net::cluster::AbstractServerNode::forPeer(
        const PeerIdentifier& peerId, ForeachPeerFunc func, void *context) {
    size_t i = 0;
    bool retval = false;

    this->peersLock.Lock();
    try {
        i = this->findPeerNode(peerId);

        try {
            func(this, peerId, this->peers[i]->Socket, context);
            retval = true;
        } catch (the::exception& e) {
            VL_DBGONLY_REFERENCED_LOCAL_VARIABLE(e);
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "ForeachPeerFunc failed for node %u "
                "(%s) with an exception: %s\n", i,
                peerId.ToStringA().PeekBuffer(), e.what());
            // TODO: second chance??????
            this->disconnectPeer(i);
        } catch (...) {
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "ForeachPeerFunc failed for node %u "
                "(%s) with a non-VISlib exception.\n", i,
                peerId.ToStringA().PeekBuffer());
            // TODO: second chance??????
            this->disconnectPeer(i);
        }
    } catch (the::no_such_element_exception e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, e.what());
    }

    this->peersLock.Unlock();
    return retval;
}


/*
 * vislib::net::cluster::AbstractServerNode::onMessageReceiverExiting
 */
void vislib::net::cluster::AbstractServerNode::onMessageReceiverExiting(
        Socket& socket, PReceiveMessagesCtx rmc) {
    // TODO remove the node.
    AbstractClusterNode::onMessageReceiverExiting(socket, rmc);
}


/*
 * vislib::net::cluster::AbstractServerNode::operator =
 */
vislib::net::cluster::AbstractServerNode& 
vislib::net::cluster::AbstractServerNode::operator =(
        const AbstractServerNode& rhs) {
    if (this != &rhs) {
        AbstractClusterNode::operator =(rhs);
        TcpServer::Listener::operator =(rhs);
        throw the::argument_exception("rhs", __FILE__, __LINE__);
    }
    return *this;
}


/*
 * vislib::net::cluster::AbstractServerNode::findPeerNode
 */
size_t vislib::net::cluster::AbstractServerNode::findPeerNode(
        const PeerIdentifier& peerId) {
    PeerIdentifier addr;

    this->peersLock.Lock();
    for (size_t i = 0; i < this->peers.Count(); i++) {
        try {
            if (this->peers[i]->Socket.GetPeerEndPoint() == peerId) {
                this->peersLock.Unlock();
                return i;
            }
        } catch (SocketException e) {
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Could not determine identifier of "
                "peer node for a client socket: %s\n", e.what());
        }
    }
    this->peersLock.Unlock();

    throw the::no_such_element_exception("The requested peer node is not known", 
        __FILE__, __LINE__);
}
