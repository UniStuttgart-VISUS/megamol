/*
 * AbstractServerNode.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/AbstractServerNode.h"

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/SocketException.h"
#include "vislib/SystemException.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"

#include "messagereceiver.h"


/*
 * vislib::net::cluster::AbstractServerNode::~AbstractServerNode
 */
vislib::net::cluster::AbstractServerNode::~AbstractServerNode(void) {
    try {
        while (this->countPeers() > 0) {
            this->disconnectPeer(0);
        }
    } catch (Exception e) {
        TRACE(Trace::LEVEL_VL_WARN, "Exception while releasing "
            "ServerNodeAdapter: %s\n", e.GetMsgA());
    } catch (...) {
        TRACE(Trace::LEVEL_VL_WARN, "Unexpected exception whie releasing "
            "ServerNodeAdapter.\n");
    }
}

/*
 * vislib::net::cluster::AbstractServerNode::GetBindAddress
 */
const vislib::net::SocketAddress& 
vislib::net::cluster::AbstractServerNode::GetBindAddress(void) const {
    return this->bindAddress;
}


/*
 * vislib::net::cluster::AbstractServerNode::Initialise
 */
void vislib::net::cluster::AbstractServerNode::Initialise(
        sys::CmdLineProviderA& inOutCmdLine) {
    // Do not call superclass method as it is not initalised.
    // TODO: Get server parameters
}


/*
 * vislib::net::cluster::AbstractServerNode::Initialise
 */
void vislib::net::cluster::AbstractServerNode::Initialise(
        sys::CmdLineProviderW& inOutCmdLine) {
    // Do not call superclass method as it is not initalised.
    // TODO: Get server parameters
}


/*
 * vislib::net::cluster::AbstractServerNode::OnNewConnection
 */
bool vislib::net::cluster::AbstractServerNode::OnNewConnection(Socket& socket,
        const SocketAddress& addr) throw() {
    try {
        PeerNode *peerNode = new PeerNode;
        peerNode->Address = addr;
        peerNode->Socket = socket;
        peerNode->Receiver = new sys::Thread(ReceiveMessages);

        ReceiveMessagesCtx *rmc = AllocateRecvMsgCtx(this, &peerNode->Socket);
        try {
            VERIFY(peerNode->Receiver->Start(static_cast<void *>(rmc)));
        } catch (Exception e) {
            FreeRecvMsgCtx(rmc);
            throw e;
        }

        this->peersLock.Lock();
        this->peers.Add(peerNode);
        this->peersLock.Unlock();
        return true;
    } catch (Exception e) {
        TRACE(Trace::LEVEL_VL_ERROR, "Could not accept peer node %s in "
            "ServerNodeAdapter because of an exception: %s\n", 
            addr.ToStringA().PeekBuffer(), e.GetMsgA());
    } catch (...) {
        TRACE(Trace::LEVEL_VL_ERROR, "Could not accept peer node %s in "
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
DWORD vislib::net::cluster::AbstractServerNode::Run(void) {
    bool isStarted = this->server.Start(&this->bindAddress);
    // TODO: generate error here for restart?
    return 0;
}


/*
 * vislib::net::cluster::AbstractServerNode::SetBindAddress
 */
void vislib::net::cluster::AbstractServerNode::SetBindAddress(
        const SocketAddress& bindAddress) {
    this->bindAddress = bindAddress;
}


/*
 * vislib::net::cluster::AbstractServerNode::SetBindAddress
 */
void vislib::net::cluster::AbstractServerNode::SetBindAddress(
        const unsigned short port) {
    this->bindAddress = SocketAddress::CreateInet(port);
}


/*
 * vislib::net::cluster::AbstractServerNode::AbstractServerNode
 */
vislib::net::cluster::AbstractServerNode::AbstractServerNode(void) 
        : AbstractClusterNode(), TcpServer::Listener(),
        bindAddress(SocketAddress::FAMILY_INET, DEFAULT_PORT) {
    this->server.GetRunnableInstance().AddListener(this);
}


/*
 * vislib::net::cluster::AbstractServerNode::AbstractServerNode
 */
vislib::net::cluster::AbstractServerNode::AbstractServerNode(
        const AbstractServerNode& rhs) 
        : AbstractClusterNode(rhs), TcpServer::Listener(rhs) {
    throw UnsupportedOperationException("AbstractServerNode", __FILE__,
        __LINE__);
}


/*
 * vislib::net::cluster::AbstractServerNode::countPeers
 */
SIZE_T vislib::net::cluster::AbstractServerNode::countPeers(void) const {
    SIZE_T retval = 0;
    this->peersLock.Lock();
    retval = this->peers.Count();
    this->peersLock.Unlock();
    return retval;
}


/*
 * vislib::net::cluster::AbstractServerNode::disconnectPeer
 */
void vislib::net::cluster::AbstractServerNode::disconnectPeer(
        const SIZE_T idx) {
    this->peersLock.Lock();

    // We assure the user that erasing non-existent peers will silently fail, 
    // so catch all exception that we expect to occur here.
    try {
        PeerNode& peerNode = *this->peers[idx];
        peerNode.Socket.Close();
        SAFE_DELETE(peerNode.Receiver);
        this->peers.Erase(idx);

    } catch (SocketException se) {
        TRACE(Trace::LEVEL_VL_WARN, "SocketException occurred when "
            "disconnecting node %u: %s\n", idx, se.GetMsgA());
    } catch (sys::SystemException sye) {
        TRACE(Trace::LEVEL_VL_WARN, "SystemException occurred when "
            "disconnecting node %u: %s\n", idx, sye.GetMsgA());
    } catch (OutOfRangeException oore) {
        TRACE(Trace::LEVEL_VL_WARN, "OutOfRangeException occurred when "
            "disconnecting node %u: %s\n", idx, oore.GetMsgA());
    }

    this->peersLock.Unlock();
}


/*
 * vislib::net::cluster::AbstractServerNode::forEachPeer
 */
SIZE_T vislib::net::cluster::AbstractServerNode::forEachPeer(
        ForeachPeerFunc func, void *context) {
    SIZE_T retval = 0;

    this->peersLock.Lock();
    for (SIZE_T i = 0; i < this->peers.Count(); i++) {
        try {
            bool isContinue = func(this, this->peers[i]->Address, 
                this->peers[i]->Socket, context);
            retval++;

            if (!isContinue) {
                break;
            }
        } catch (Exception& e) {
            TRACE(Trace::LEVEL_VL_WARN, "ForeachPeerFunc failed for node %u "
                "with an exception: %s\n", i, e.GetMsgA());
            // TODO: second chance??????
            this->peers.Erase(i);
            i--;
        } catch (...) {
            TRACE(Trace::LEVEL_VL_WARN, "ForeachPeerFunc failed for node %u "
                "with a non-VISlib exception.\n", i);
            // TODO: second chance??????
            this->peers.Erase(i);
            i--;
        }
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
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }
    return *this;
}
