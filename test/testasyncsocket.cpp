/*
 * testasyncsocket.cpp
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifdef _WIN32
#include <winsock2.h>
#endif /* _WIN32 */

#include "testasyncsocket.h"
#include "testhelper.h"

#include "vislib/AsyncSocket.h"
#include "vislib/AsyncSocketContext.h"
#include "vislib/RunnableThread.h"


using namespace vislib::net;


static const unsigned short TEST_PORT = 65432;


////////////////////////////////////////////////////////////////////////////////
// Client and sender thread.

class Sender : public vislib::sys::Runnable {
public:
    inline Sender(void) {}
    virtual ~Sender(void);
    virtual DWORD Run(void *userData);
};

Sender::~Sender(void) {}

DWORD Sender::Run(void *userData) {
    UINT_PTR cnt = reinterpret_cast<UINT_PTR>(userData);
    AsyncSocketContext ctx;

    try {
        AssertNoException("Socket::Startup", Socket::Startup());

        AsyncSocket socket;
        AssertNoException("Create client socket", socket.Create(
            Socket::FAMILY_INET, Socket::TYPE_STREAM, Socket::PROTOCOL_TCP));

        IPEndPoint endPoint(IPAddress::LOCALHOST, TEST_PORT);
        AssertNoException("Connect to server", socket.Connect(endPoint));

        for (UINT_PTR i = 0; i < cnt; i++) {
            ctx.Reset();
            socket.BeginSend(&i, sizeof(i), &ctx);
            ctx.Wait();
            AssertEqual("Sent sufficient data", socket.EndSend(&ctx),
                static_cast<SIZE_T>(sizeof(UINT_PTR)));
        }

        AssertNoException("Socket::Cleanup", Socket::Cleanup());

        return cnt;
    } catch (...) {
        return 0;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Server and receiver thread.

class Receiver : public vislib::sys::Runnable {
public:
    inline Receiver(void) {}
    virtual ~Receiver(void);
    virtual DWORD Run(void *userData);
};

Receiver::~Receiver(void) {}

DWORD Receiver::Run(void *userData) {
    UINT_PTR cnt = reinterpret_cast<UINT_PTR>(userData);
    UINT_PTR data;
    AsyncSocketContext ctx;

    try {
        AssertNoException("Socket::Startup", Socket::Startup());

        Socket serverSocket;
        AssertNoException("Create server socket", serverSocket.Create(
            Socket::FAMILY_INET, Socket::TYPE_STREAM, Socket::PROTOCOL_TCP));

        IPEndPoint endPoint(IPAddress::ANY, TEST_PORT);
        AssertNoException("Bind server socket", serverSocket.Bind(endPoint));
        
        AssertNoException("Listen on server socket", serverSocket.Listen());
        AsyncSocket socket(serverSocket.Accept());

        for (UINT_PTR i = 0; i < cnt; i++) {
            ctx.Reset();
            socket.BeginReceive(&data, sizeof(data), &ctx);
            ctx.Wait();
            AssertEqual("Received sufficient data", socket.EndReceive(&ctx),
                static_cast<SIZE_T>(sizeof(UINT_PTR)));
        }

        AssertNoException("Socket::Cleanup", Socket::Cleanup());

        return cnt;
    } catch (...) {
        return 0;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Test function.

void TestAsyncSocket(void) {
    vislib::sys::RunnableThread<Sender> sender;
    vislib::sys::RunnableThread<Receiver> receiver;

    UINT_PTR cnt = 10;
    sender.Start(reinterpret_cast<void *>(cnt));
    receiver.Start(reinterpret_cast<void *>(cnt));

    sender.Join();
    receiver.Join();
}
