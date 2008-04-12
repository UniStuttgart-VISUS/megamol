/*
 * ClientNodeAdapter.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/ClientNodeAdapter.h"

#include "vislib/assert.h"
#include "vislib/CmdLineParser.h"
#include "vislib/IllegalStateException.h"
#include "vislib/memutils.h"
#include "vislib/SocketException.h"
#include "vislib/Trace.h"

#include "messagereceiver.h"


/*
 * vislib::net::cluster::ClientNodeAdapter::~ClientNodeAdapter
 */
vislib::net::cluster::ClientNodeAdapter::~ClientNodeAdapter(void) {
    try {
        this->socket.Close();
        SAFE_DELETE(this->receiver);
        Socket::Cleanup();
    } catch (Exception e) {
        TRACE(Trace::LEVEL_VL_WARN, "Exception while releasing "
            "ClientNodeAdapter: %s\n", e.GetMsgA());
    } catch (...) {
        TRACE(Trace::LEVEL_VL_WARN, "Unexpected exception whie releasing "
            "ClientNodeAdapter.\n");
    }
}


/*
 * vislib::net::cluster::ClientNodeAdapter::GetServerAddress
 */
const vislib::net::SocketAddress& 
vislib::net::cluster::ClientNodeAdapter::GetServerAddress(void) const {
    return this->serverAddress;
}


/*
 * vislib::net::cluster::ClientNodeAdapter::Initialise
 */
void vislib::net::cluster::ClientNodeAdapter::Initialise(
        sys::CmdLineProviderA& inOutCmdLine) {
    // Do not call superclass method as it is not initalised.
    // TODO: Get client parameters

    typedef vislib::sys::CmdLineParserA Parser;
    typedef Parser::Option Option;
    typedef Option::ValueDesc ValueDesc;


    Parser parser;

    Option optServer("server-address", 
        "Specifies the address of the server node.",
        Option::FLAG_UNIQUE, 
        ValueDesc::ValueList(Option::STRING_VALUE, 
        "server", "Specifies the server host name or IP address.")
        ->Add(Option::INT_VALUE, 
        "port", "Specifies the server port."));
    parser.AddOption(&optServer);


}


/*
 * vislib::net::cluster::ClientNodeAdapter::Initialise
 */
void vislib::net::cluster::ClientNodeAdapter::Initialise(
        sys::CmdLineProviderW& inOutCmdLine) {
    // Do not call superclass method as it is not initalised.
    // TODO: Get client parameters
}


/*
 * vislib::net::cluster::ClientNodeAdapter::Run
 */
DWORD vislib::net::cluster::ClientNodeAdapter::Run(void) {
    if (this->socket.IsValid() || (this->receiver != NULL)) {
        throw IllegalStateException("ClientNodeAdapter::Run can only be called "
            "once for connecting to the server node.", __FILE__, __LINE__);
    }

    /* Clean up if in some case of illegal state. */
    SAFE_DELETE(this->receiver);
    try {
        this->socket.Close();
    } catch (...) {
    }

    /* Connect to the server. */
    this->socket.Create(Socket::FAMILY_INET, Socket::TYPE_STREAM, 
        Socket::PROTOCOL_TCP);
    this->socket.Connect(this->serverAddress);

    /* Start the message receiver. */
    this->receiver = new sys::Thread(ReceiveMessages);

    ReceiveMessagesCtx *rmc = new ReceiveMessagesCtx;
    rmc->Receiver = this;
    rmc->Socket = &this->socket;

    try {
        VERIFY(this->receiver->Start(static_cast<void *>(rmc)));
    } catch (Exception e) {
        SAFE_DELETE(rmc);
        throw e;
    }

    return 0;
}


/*
 * vislib::net::cluster::ClientNodeAdapter::SetServerAddress
 */
void vislib::net::cluster::ClientNodeAdapter::SetServerAddress(
        const SocketAddress& serverAddress) {
    this->serverAddress = serverAddress;
}


/*
 * vislib::net::cluster::ClientNodeAdapter::ClientNodeAdapter
 */
vislib::net::cluster::ClientNodeAdapter::ClientNodeAdapter(void) 
        : Super(), receiver(NULL) {
    try {
        Socket::Startup();
    } catch (SocketException e) {
        TRACE(Trace::LEVEL_VL_ERROR, "Socket::Startup failed in "
            "ClientNodeAdapter::ctor. The instance will probably not work. "
            "Details: %s\n", e.GetMsgA());
    }
}


/*
 * vislib::net::cluster::ClientNodeAdapter::ClientNodeAdapter
 */
vislib::net::cluster::ClientNodeAdapter::ClientNodeAdapter(
        const ClientNodeAdapter& rhs) 
        : Super(rhs), receiver(NULL) {
    try {
        Socket::Startup();
    } catch (SocketException e) {
        TRACE(Trace::LEVEL_VL_ERROR, "Socket::Startup failed in "
            "ClientNodeAdapter::ctor. The instance will probably not work. "
            "Details: %s\n", e.GetMsgA());
    }
}


/*
 * vislib::net::cluster::ClientNodeAdapter::countPeers
 */
SIZE_T vislib::net::cluster::ClientNodeAdapter::countPeers(void) const {
    return (this->socket.IsValid() ? 1 : 0);
}


/*
 * vislib::net::cluster::ClientNodeAdapter::forEachPeer
 */
SIZE_T vislib::net::cluster::ClientNodeAdapter::forEachPeer(
        ForeachPeerFunc func, void *context) {
    SIZE_T retval = 0;

    try {
        func(this, this->serverAddress, this->socket, context);
        retval = 1;
    } catch (Exception e) {
        TRACE(Trace::LEVEL_VL_WARN, "ForeachPeerFunc failed "
            "with an exception: %s", e.GetMsgA());
    } catch (...) {
        TRACE(Trace::LEVEL_VL_WARN, "ForeachPeerFunc failed "
            "with a non-VISlib exception.");
    }

    return retval;
}


/*
 * vislib::net::cluster::ClientNodeAdapter::operator =
 */
vislib::net::cluster::ClientNodeAdapter& 
vislib::net::cluster::ClientNodeAdapter::operator =(
        const ClientNodeAdapter& rhs) {
    Super::operator =(rhs);
    return *this;
}
