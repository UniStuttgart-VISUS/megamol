/*
 * testcomm.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/TcpCommChannel.h"  // Must be first
#include "testcomm.h"

#include "testhelper.h"
#include "vislib/Event.h"
#include "vislib/SmartRef.h"
#include "vislib/String.h"
#include "vislib/RunnableThread.h"
#include "vislib/StringConverter.h"
#include "vislib/IPCommEndPoint.h"


static vislib::sys::Event evtServerBound(true);


class Worker : public vislib::sys::Runnable {
public:
    virtual DWORD Run(void *userData);

    vislib::SmartRef<vislib::net::AbstractCommEndPoint> EndPoint;
};

DWORD Worker::Run(void *userData) {
    using namespace vislib::net;

    Socket::Startup();

    bool isServer = (userData != 0);
    int data;
    vislib::SmartRef<TcpCommChannel> comm = TcpCommChannel::Create();
    vislib::SmartRef<TcpCommChannel> client;

    try {
        if (isServer) {
            std::cout << "Server binding to " << this->EndPoint->ToStringA().PeekBuffer() << " ..." << std::endl;
            comm->Bind(this->EndPoint);
            comm->Listen();
            std::cout << "Server (" << &(*comm) << ") ready and waiting now ..." << std::endl;
            ::evtServerBound.Set(); // This is actually not totally safe, but enough for a test.
            client = comm->Accept().DynamicCast<TcpCommChannel>();
            std::cout << "Client acceppted " << client->GetSocket().GetPeerEndPoint().ToStringA().PeekBuffer() << "." << std::endl;
            comm->Close();

            client->Receive(&data, sizeof(data));
            ::AssertEqual("Received 12", data, 12);

        } else {
            std::cout << "Client connecting to " << this->EndPoint->ToStringA().PeekBuffer()  << " ..." << std::endl;
            comm->Connect(this->EndPoint);
            std::cout << "Client (" << &(*comm) << ") connected to " << comm->GetSocket().GetPeerEndPoint().ToStringA().PeekBuffer() << "." << std::endl;

            data = 12;
            comm->Send(&data, sizeof(data));
        }
    } catch (vislib::Exception e) {
        std::cerr << e.GetMsgA() << std::endl;
        throw e;
    }

    Socket::Cleanup();
    return 0;
}


void TestIpCommEndPoint(void) {
    using namespace vislib;
    using namespace vislib::net;
    SmartRef<AbstractCommEndPoint> a;

    a = IPCommEndPoint::Create("localhost:2222");
    std::cout << "localhost:2222" << " -> " << a->ToStringA().PeekBuffer() << std::endl;

    a = IPCommEndPoint::Create("127.0.0.1:2222");
    std::cout << "127.0.0.1:2222" << " -> " << a->ToStringA().PeekBuffer() << std::endl;

    a = IPCommEndPoint::Create(IPCommEndPoint::IPV4, "www.microsoft.com", 80);
    std::cout << "IPV4, www.microsoft.com, 80" << " -> " << a->ToStringA().PeekBuffer() << std::endl;
}


void TestComm(void) {
    using namespace vislib;
    using namespace vislib::net;

    Socket::Startup();

    ::TestIpCommEndPoint();

    vislib::sys::RunnableThread<Worker> server;
    server.EndPoint = IPCommEndPoint::Create(IPCommEndPoint::IPV4, 12345);

    vislib::sys::RunnableThread<Worker> client;
    client.EndPoint = IPCommEndPoint::Create(IPCommEndPoint::IPV4, "127.0.0.1:12345");

    ::evtServerBound.Reset();
    server.Start((void *) true);
    ::evtServerBound.Wait();
    client.Start((void *) false);

    server.Join();
    client.Join();

    Socket::Cleanup();
}
