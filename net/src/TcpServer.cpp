/*
 * TcpServer.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/TcpServer.h"

#include "vislib/IllegalParamException.h"
#include "vislib/Interlocked.h"
#include "vislib/SocketException.h"
#include "the/trace.h"



/*
 * vislib::net::TcpServer::Listener::~Listener
 */
vislib::net::TcpServer::Listener::~Listener(void) {
    // Nothing to to.
}


/*
 * vislib::net::TcpServer::Listener::Listener
 */
vislib::net::TcpServer::Listener::Listener(void) {
    // Nothing to to.
}


/*
 * vislib::net::TcpServer::Listener::OnServerStopped
 */
void vislib::net::TcpServer::Listener::OnServerStopped(void) throw() {
    // Nothing to do.
}


/*
 * vislib::net::TcpServer::FLAGS_REUSE_ADDRESS
 */
const uint32_t vislib::net::TcpServer::FLAGS_REUSE_ADDRESS = 0x0002;


/*
 * vislib::net::TcpServer::FLAGS_SHARE_ADDRESS
 */
const uint32_t vislib::net::TcpServer::FLAGS_SHARE_ADDRESS = 0x0001;


/*
 * vislib::net::TcpServer::TcpServer
 */
vislib::net::TcpServer::TcpServer(const uint32_t flags) : flags(flags) {
    // Nothing to do.
}


/*
 * vislib::net::TcpServer::~TcpServer
 */
vislib::net::TcpServer::~TcpServer(void) {
    this->Terminate();
}


/*
 * vislib::net::TcpServer::AddListener
 */
void vislib::net::TcpServer::AddListener(Listener *listener) {
    if (listener != NULL) {
        this->lock.Lock();
        this->listeners.Append(listener);
        this->lock.Unlock();
    }
}


/*
 * vislib::net::TcpServer::RemoveListener
 */
void vislib::net::TcpServer::RemoveListener(Listener *listener) {
    if (listener != NULL) {
        this->lock.Lock();
        this->listeners.RemoveAll(listener);
        this->lock.Unlock();
    }
}


/*
 * vislib::net::TcpServer::Run
 */
unsigned int vislib::net::TcpServer::Run(void *userData) {
    THE_ASSERT(userData != NULL);
    if (userData != NULL) {
        IPEndPoint serverAddr = *static_cast<IPEndPoint *>(userData);
        return this->Run(serverAddr);
    } else {
        throw IllegalParamException("userData", __FILE__, __LINE__);
    }
}


/*
 * vislib::net::TcpServer::Run
 */
unsigned int vislib::net::TcpServer::Run(const IPEndPoint& serverAddr) {
    IPEndPoint peerAddr;
    Socket peerSocket;
    unsigned int retval = 0;

    /* Prepare the socket subsystem */
    try {
        Socket::Startup();
    } catch (SocketException e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Socket::Startup in TcpServer failed: "
            "%s\n", e.GetMsgA());
        return e.GetErrorCode();
    }

    /* Clean existing resources, if any. */
    try {
        if (this->socket.IsValid()) {
            this->socket.Close();
        }
    } catch (SocketException e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Error while cleaning existing socket of "
            "TcpServer: %s\n", e.GetMsgA());
    }

    /* Create socket and bind it to specified address. */
    try {
        this->socket.Create(serverAddr, Socket::TYPE_STREAM, 
            Socket::PROTOCOL_TCP);
        if (this->IsSharingAddress()) {
            // TODO: Problems with admin rights
            this->socket.SetExclusiveAddrUse(false);
        }
        if (this->IsReuseAddress()) {
            this->socket.SetReuseAddr(true);
        }
        this->socket.Bind(serverAddr);
    } catch (SocketException e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_ERROR, "Creating or binding server socket of "
            "TcpServer server: %s\n", e.GetMsgA());
        retval = e.GetErrorCode();
    }

    /* Enter server loop if no error so far. */
    if (retval == 0) {
        try {
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "The TcpServer is listening on "
                "%s ...\n", serverAddr.ToStringA().PeekBuffer());

            while (true) {
                this->socket.Listen();
                peerSocket = this->socket.Accept(&peerAddr);
                THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "TcpServer accepted new connection "
                    "from %s.\n", peerAddr.ToStringA().PeekBuffer());

                if (!this->fireNewConnection(peerSocket, peerAddr)) {
                    THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_INFO, "TcpServer is closing "
                        "connection to %s because no one is interested in this "
                        "client.\n", peerAddr.ToStringA().PeekBuffer());
                    try {
                        peerSocket.Close();
                    } catch (SocketException e) {
                        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Closing unused peer "
                            "connection: %s\n", e.GetMsgA());
                    }
                }
            }
        } catch (SocketException e) {
            THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Communication error in TcpServer: "
                "%s\n", e.GetMsgA());
        }
    }

    /* Inform listener that server exits. */
    this->fireServerStopped();

    /* Clean up socket library. */
    this->Terminate();
    try {
        Socket::Cleanup();
    } catch (SocketException e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "Error during TcpServer shutdown: %s\n",
            e.GetMsgA());
        retval = e.GetErrorCode();
    }

    return retval;
}


/*
 * vislib::net::TcpServer::SetFlags
 */
void vislib::net::TcpServer::SetFlags(uint32_t flags) {
    vislib::sys::Interlocked::Exchange(reinterpret_cast<int32_t*>(&this->flags),
        static_cast<int32_t>(flags));
}


/*
 * vislib::net::TcpServer::Terminate
 */
bool vislib::net::TcpServer::Terminate(void) {
    try {
        this->socket.Shutdown();
    } catch (SocketException e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "SocketException when shutting "
            "TcpServer down: %s\n", e.GetMsgA());
    }
    try {
        this->socket.Close();
    } catch (SocketException e) {
        THE_TRACE(THE_TRCCHL_DEFAULT, THE_TRCLVL_WARN, "SocketException when terminating "
            "TcpServer: %s\n", e.GetMsgA());
    }
    return true;
}


/*
 * vislib::net::TcpServer::fireNewConnection
 */
bool vislib::net::TcpServer::fireNewConnection(Socket& socket, 
         const IPEndPoint& addr) {
    bool retval = false;

    this->lock.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        if ((retval = it.Next()->OnNewConnection(socket, addr))) {
            break;
        }
    }

    this->lock.Unlock();
    return retval;
}


/*
 * vislib::net::TcpServer::fireServerStopped
 */
void vislib::net::TcpServer::fireServerStopped(void) {
    this->lock.Lock();
    ListenerList::Iterator it = this->listeners.GetIterator();
    while (it.HasNext()) {
        it.Next()->OnServerStopped();
    }
    this->lock.Unlock();
}
