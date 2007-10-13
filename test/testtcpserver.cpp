/*
 * testtcpserver.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "testtcpserver.h"

#include "vislib/TcpServer.h"
#include "vislib/SocketException.h"
#include "vislib/SystemException.h"
#include "vislib/Thread.h"
#include "testhelper.h"

#include <iostream>

/** Port used for test server. */
static const SHORT TEST_PORT = 8665;


class ConnectionListener : public vislib::net::TcpServer::Listener {

public:
    
    inline ConnectionListener(void) : cntConnections(0) {}
    virtual ~ConnectionListener(void);
    virtual bool OnNewConnection(vislib::net::Socket& socket, 
        const vislib::net::SocketAddress& addr) throw();
    virtual void OnServerStopped(void) throw();

    int cntConnections;
};

ConnectionListener::~ConnectionListener(void) {
}

bool ConnectionListener::OnNewConnection(vislib::net::Socket& socket,
        const vislib::net::SocketAddress& addr) throw() {
    std::cout << "The TCP server accepted the connection from "
        << addr.ToStringA().PeekBuffer() << std::endl;
    this->cntConnections++;
    return false;
}

void ConnectionListener::OnServerStopped(void) throw() {
    std::cout << "The TCP server exits ..." << std::endl;
}


void TestTcpServer(void) {
    using namespace vislib::net;
    using namespace vislib::sys;

    ConnectionListener cl1, cl2, cl3;
    TcpServer server;
    Thread serverThread(&server);
    SocketAddress serverAddr = SocketAddress::CreateInet("127.0.0.1", 
        TEST_PORT);

    server.AddListener(&cl1);
    server.AddListener(&cl2);
    server.AddListener(&cl3);

    serverThread.Start(&serverAddr);

    AssertNoException("Socket startup", Socket::Startup());

    Socket socket;
    AssertNoException("Create client socket", socket.Create(
        Socket::FAMILY_INET, Socket::TYPE_STREAM, Socket::PROTOCOL_TCP));

    AssertNoException("Connecting client socket", socket.Connect(serverAddr));

    serverThread.Terminate(false);

    AssertEqual("Listener 1 was informed\n", cl1.cntConnections, 1);
    AssertEqual("Listener 2 was informed\n", cl2.cntConnections, 1);
    AssertEqual("Listener 3 was informed\n", cl3.cntConnections, 1);

    AssertNoException("Socket cleanup", Socket::Cleanup());
}
