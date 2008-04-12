/*
 * ServerNodeAdapter.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/ServerNodeAdapter.h"

#include "vislib/IllegalParamException.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/SocketException.h"
#include "vislib/SystemException.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"

#include "messagereceiver.h"


/*
 * vislib::net::cluster::ServerNodeAdapter::~ServerNodeAdapter
 */
vislib::net::cluster::ServerNodeAdapter::~ServerNodeAdapter(void) {
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
 * vislib::net::cluster::ServerNodeAdapter::GetBindAddress
 */
const vislib::net::SocketAddress& 
vislib::net::cluster::ServerNodeAdapter::GetBindAddress(void) const {
    return this->bindAddress;
}


/*
 * vislib::net::cluster::ServerNodeAdapter::Initialise
 */
void vislib::net::cluster::ServerNodeAdapter::Initialise(
        sys::CmdLineProviderA& inOutCmdLine) {
    // Do not call superclass method as it is not initalised.
    // TODO: Get server parameters
}


/*
 * vislib::net::cluster::ServerNodeAdapter::Initialise
 */
void vislib::net::cluster::ServerNodeAdapter::Initialise(
        sys::CmdLineProviderW& inOutCmdLine) {
    // Do not call superclass method as it is not initalised.
    // TODO: Get server parameters
}


/*
 * vislib::net::cluster::ServerNodeAdapter::OnNewConnection
 */
bool vislib::net::cluster::ServerNodeAdapter::OnNewConnection(Socket& socket,
        const SocketAddress& addr) throw() {
    try {
        PeerNode *peerNode = new PeerNode;
        peerNode->Address = addr;
        peerNode->Socket = socket;
        peerNode->Receiver = new sys::Thread(ReceiveMessages);

        ReceiveMessagesCtx rmc;
        rmc.Receiver = this;
        rmc.Socket = peerNode->Socket;

        peerNode->Receiver->Start(static_cast<void *>(&rmc));

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
 * vislib::net::cluster::ServerNodeAdapter::OnServerStopped
 */
void vislib::net::cluster::ServerNodeAdapter::OnServerStopped(void) throw() {
}


/*
 * vislib::net::cluster::ServerNodeAdapter::Run
 */
DWORD vislib::net::cluster::ServerNodeAdapter::Run(void) {
    bool isStarted = this->server.Start(&this->bindAddress);
    // TODO: generate error here for restart?
    return 0;
}


/*
 * vislib::net::cluster::ServerNodeAdapter::SetBindAddress
 */
void vislib::net::cluster::ServerNodeAdapter::SetBindAddress(
        const SocketAddress& bindAddress) {
    this->bindAddress = bindAddress;
}


/*
 * vislib::net::cluster::ServerNodeAdapter::SetBindAddress
 */
void vislib::net::cluster::ServerNodeAdapter::SetBindAddress(
        const unsigned short port) {
    this->bindAddress = SocketAddress::CreateInet(port);
}


/*
 * vislib::net::cluster::ServerNodeAdapter::ServerNodeAdapter
 */
vislib::net::cluster::ServerNodeAdapter::ServerNodeAdapter(void) 
        : AbstractServerNode() {
    this->server.GetRunnableInstance().AddListener(this);
}


/*
 * vislib::net::cluster::ServerNodeAdapter::ServerNodeAdapter
 */
vislib::net::cluster::ServerNodeAdapter::ServerNodeAdapter(
        const ServerNodeAdapter& rhs) : AbstractServerNode(rhs) {
    throw UnsupportedOperationException("ServerNodeAdapter", __FILE__, 
        __LINE__);
}


/*
 * vislib::net::cluster::ServerNodeAdapter::countPeers
 */
SIZE_T vislib::net::cluster::ServerNodeAdapter::countPeers(void) const {
    SIZE_T retval = 0;
    this->peersLock.Lock();
    retval = this->peers.Count();
    this->peersLock.Unlock();
    return retval;
}


/*
 * vislib::net::cluster::ServerNodeAdapter::disconnectPeer
 */
void vislib::net::cluster::ServerNodeAdapter::disconnectPeer(const SIZE_T idx) {
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
 * vislib::net::cluster::ServerNodeAdapter::forEachPeer
 */
SIZE_T vislib::net::cluster::ServerNodeAdapter::forEachPeer(
        ForeachPeerFunc func, void *context) {
    SIZE_T retval = 0;

    this->peersLock.Lock();
    for (SIZE_T i = 0; i < this->peers.Count(); i++) {
        try {
            bool isContinue = func(dynamic_cast<AbstractServerNode *>(this),
                this->peers[i]->Address, this->peers[i]->Socket, context);
            retval++;

            if (!isContinue) {
                break;
            }
        } catch (Exception e) {
            TRACE(Trace::LEVEL_VL_WARN, "ForeachPeerFunc failed for node %u "
                "with an exception: %s", i, e.GetMsgA());
        } catch (...) {
            TRACE(Trace::LEVEL_VL_WARN, "ForeachPeerFunc failed for node %u "
                "with a non-VISlib exception.", i);
        }
    }
    this->peersLock.Unlock();

    return retval;
}


/*
 * vislib::net::cluster::ServerNodeAdapter::operator =
 */
vislib::net::cluster::ServerNodeAdapter& 
vislib::net::cluster::ServerNodeAdapter::operator =(
        const ServerNodeAdapter& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    AbstractServerNode::operator =(rhs);
    return *this;
}
