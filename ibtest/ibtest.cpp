/*
 * ibtest.cpp
 *
 * Copyright (C) 2012 by Visualisierungsinstitut der Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "vislib/COMException.h"
#include "vislib/Event.h"
#include "vislib/IbvCommChannel.h"
#include "vislib/IbvCommServerChannel.h"
#include "vislib/IbRdmaCommClientChannel.h"
#include "vislib/IbRdmaCommServerChannel.h"
#include "vislib/IbRdmaException.h"
#include "vislib/IbvInformation.h"
#include "vislib/IPCommEndPoint.h"
#include "vislib/RunnableThread.h"
#include "vislib/Trace.h"


typedef vislib::SmartRef<vislib::net::AbstractCommEndPoint> IbEndPoint;
typedef vislib::SmartRef<vislib::net::ib::IbRdmaCommClientChannel> IbChannel;
typedef vislib::SmartRef<vislib::net::ib::IbRdmaCommServerChannel> IbServerChannel;


static const char REFERENCE_DATA[] = { "Hier spricht der Hugo!" };

static vislib::sys::Event evtServerReady(true);

static const int CNT_CLIENTS = 2;


class Server : public vislib::sys::Runnable {
public:
    inline Server(void) { }
    virtual DWORD Run(void *userData);
};

DWORD Server::Run(void *userData) {
    using namespace vislib;
    using namespace vislib::net;
    using namespace vislib::net::ib;

    try {
        IbServerChannel channel = IbRdmaCommServerChannel::Create(512);

        //IbEndPoint ep = IPCommEndPoint::Create(IPEndPoint::FAMILY_INET,
        //    "192.168.219.250", 12345);
        IbEndPoint ep = IPCommEndPoint::Create(IPEndPoint::FAMILY_INET, 12345);

        std::cout << "Bind..." << std::endl;
        channel->Bind(ep);

        std::cout << "Listen..." << std::endl;
        channel->Listen();

        ::evtServerReady.Set();

        std::cout << "Server is bound to " 
            << channel->GetLocalEndPoint()->ToStringA().PeekBuffer() 
            << std::endl;

        char *data = new char[sizeof(REFERENCE_DATA)];
        for (int i = 0; i < CNT_CLIENTS; i++) {
            std::cout << "Accept..." << std::endl;
            SmartRef<AbstractCommClientChannel> client = channel->Accept();
            client->Receive(data, sizeof(REFERENCE_DATA));
            std::cout << "Received \"" << data << "\"" << std::endl;
        }
        delete[] data;

        //channel->Close();
        std::cout << "Server leaving..." << std::endl;
        return 0;

    } catch (IbRdmaException e) {
        std::cerr << "IB server failed: " << e.GetMsgA() << std::endl;
        return static_cast<DWORD>(e.GetErrorCode());
    }
}


class Client : public vislib::sys::Runnable {
public:
    inline Client(void) { }
    virtual DWORD Run(void *userData);
};

DWORD Client::Run(void *userData) {
    using namespace vislib;
    using namespace vislib::net;
    using namespace vislib::net::ib;

    try {
        IbChannel channel = IbRdmaCommClientChannel::Create(512);

        IbEndPoint ep = IPCommEndPoint::Create(IPEndPoint::FAMILY_INET,
            //"192.168.219.231", 12345);
            "192.168.219.250", 12345);

        std::cout << "Waiting for server to become available..." << std::endl;
        ::evtServerReady.Wait();

        std::cout << "Connect..." << std::endl;
        channel->Connect(ep);

        std::cout << "Client is connected to " 
            << channel->GetRemoteEndPoint()->ToStringA().PeekBuffer() 
            << " using the local end point " 
            << channel->GetLocalEndPoint()->ToStringA().PeekBuffer() 
            << std::endl;

        channel->Send(REFERENCE_DATA, sizeof(REFERENCE_DATA));


        std::cout << "Client leaving..." << std::endl;
        return 0;
    } catch (IbRdmaException e) {
        std::cerr << "IB client failed: " << e.GetMsgA() << ", " << e.GetErrorCode() << std::endl;
        return static_cast<DWORD>(e.GetErrorCode());
    }
}



int _tmain(int argc, _TCHAR **argv) {
    using namespace vislib;
    using namespace vislib::net;
    using namespace vislib::net::ib;
    using namespace vislib::sys;

    vislib::Trace::GetInstance().SetLevel(Trace::LEVEL_ALL);

    IbvInformation::DeviceList devices;

    try {
        IbvInformation::GetInstance().GetDevices(devices);

        for (SIZE_T i = 0; i < devices.Count(); i++) {
            std::cout << "Device #" << i << ":" << std::endl;

            std::cout << "\tGUID: " 
                << devices[i].GetNodeGuidA().PeekBuffer() 
                << ": " << std::endl;

            std::cout << "\tSystem Image GUID: " 
                << devices[i].GetSystemImageGuidA().PeekBuffer() << std::endl;

            std::cout << "\tNumber of ports: " << devices[i].GetPortCount() 
                << std::endl;

            for (SIZE_T j = 0; j < devices[i].GetPortCount(); j++) {
                const IbvInformation::Port& port = devices[i].GetPort(j);
                std::cout << "\tPort #" << j << ": " << std::endl;

                std::cout << "\t\tGUID: "
                    << port.GetPortGuidA().PeekBuffer()
                    << std::endl;

                std::cout << "\t\tState: " << port.GetStateA().PeekBuffer()
                    << std::endl;

                std::cout << "\t\tPhysical State: " 
                    << port.GetPhysicalStateA().PeekBuffer()
                    << std::endl;
            }
        }

    } catch (IbRdmaException e) {
        std::cerr << "Retrieving IB devices failed: " << e.GetMsgA() 
            << std::endl;
    }

    RunnableThread<Client> clientThread;
    //RunnableThread<Client> clientThread2;
    RunnableThread<Server> serverThread;

    serverThread.Start();
    clientThread.Start();
    //clientThread2.Start();

    clientThread.Join();
    serverThread.Join();
    //clientThread2.Join();

    try {
        //IbChannel channel = IbvCommChannel::Create();
        //IbChannel channel2 = IbvCommChannel::Create();
        //IbServerChannel sChannel = IbvCommServerChannel::Create();


        ////SmartRef<AbstractCommEndPoint> ep = IPCommEndPoint::Create(IPEndPoint::FAMILY_INET, 12345);
        //SmartRef<AbstractCommEndPoint> ep = IPCommEndPoint::Create(IPEndPoint::FAMILY_INET, "192.168.219.250", 12345);

        ////channel2->Connect(ep);

        //sChannel->Bind(ep);

        //clientThread.Start(NULL);

        //sChannel->Listen();
        //sChannel->Accept();
        

        //WV_DEVICE_ADDRESS dev;
        //IPCommEndPoint *e1 = ep.DynamicPeek<IPCommEndPoint>();
        //IPEndPoint& e2 = static_cast<IPEndPoint&>(*e1);
        //HRESULT hr = channel->GetProvider()->TranslateAddress(static_cast<struct sockaddr *>(e2), &dev);

        //channel->Bind(ep);
        //channel->Listen();
        //channel->Accept();
    } catch (COMException e) {
        std::cerr << "Starting IB server failed: " << e.GetMsgA() 
            << std::endl;
    }

    ::_getch();
    return 0;
}
