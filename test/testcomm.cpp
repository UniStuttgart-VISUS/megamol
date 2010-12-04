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
#include "vislib/IPCommEndPointAddress.h"


static vislib::sys::Event evtServerBound(true);


class Worker : public vislib::sys::Runnable {
public:
    virtual DWORD Run(void *userData);

    vislib::StringW Address;
};

DWORD Worker::Run(void *userData) {
    using namespace vislib::net;

    Socket::Startup();

    bool isServer = (userData != 0);
    int data;
    vislib::SmartRef<TcpCommChannel> comm(new TcpCommChannel(), false);
    vislib::SmartRef<TcpCommChannel> client;

    try {
        if (isServer) {
            std::cout << "Server binding to " << W2A(this->Address.PeekBuffer()) << " ..." << std::endl;
            comm->Bind(this->Address);
            comm->Listen();
            std::cout << "Server (" << &(*comm) << ") ready and waiting now ..." << std::endl;
            ::evtServerBound.Set(); // This is actually not totally safe, but enough for a test.
            client = comm->Accept().DynamicCast<TcpCommChannel>();
            std::cout << "Client acceppted " << client->GetSocket().GetPeerEndPoint().ToStringA().PeekBuffer() << "." << std::endl;
            comm->Close();

            client->Receive(&data, sizeof(data));
            ::AssertEqual("Received 12", data, 12);

        } else {
            std::cout << "Client connecting to " << W2A(this->Address.PeekBuffer()) << " ..." << std::endl;
            comm->Connect(this->Address);
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


void TestIpCommEndPointAddress(void) {
    using namespace vislib;
    using namespace vislib::net;
    AbstractCommEndPointAddress *a;

    a = IPCommEndPointAddress::Create("localhost:2222");
    std::cout << "localhost:2222" << " -> " << a->ToStringA().PeekBuffer() << std::endl;
    a->Release();

    a = IPCommEndPointAddress::Create("127.0.0.1:2222");
    std::cout << "127.0.0.1:2222" << " -> " << a->ToStringA().PeekBuffer() << std::endl;
    a->Release();

    a = IPCommEndPointAddress::Create(IPCommEndPointAddress::IPV4, "www.microsoft.com", 80);
    std::cout << "IPV4, www.microsoft.com, 80" << " -> " << a->ToStringA().PeekBuffer() << std::endl;
    a->Release();
}


void TestComm(void) {
    using namespace vislib;
    using namespace vislib::net;

    Socket::Startup();

    ::TestIpCommEndPointAddress();

    vislib::sys::RunnableThread<Worker> server;
    server.Address = L"0.0.0.0:12345";

    vislib::sys::RunnableThread<Worker> client;
    client.Address = L"127.0.0.1:12345";

    ::evtServerBound.Reset();
    server.Start((void *) true);
    ::evtServerBound.Wait();
    client.Start((void *) false);

    server.Join();
    client.Join();

    Socket::Cleanup();
}
