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
#include "vislib/Trace.h"

using namespace vislib::net;
using namespace vislib::sys;

// Ensure that a whole line is output at once on cout.
static CriticalSection COUT_IO_LOCK;
#define LOCK_COUT ::COUT_IO_LOCK.Lock()
#define UNLOCK_COUT ::COUT_IO_LOCK.Unlock()

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
            LOCK_COUT;
            std::cout << "BeginSend #" << i << std::endl;
            UNLOCK_COUT;
            ctx.Reset();
            socket.BeginSend(&i, sizeof(i), &ctx);

            LOCK_COUT;
            std::cout << "Wait (send) #" << i << std::endl;
            UNLOCK_COUT;
            ctx.Wait();

            LOCK_COUT;
            std::cout << "EndSend #" << i << std::endl;
            UNLOCK_COUT;
            AssertEqual("Sent sufficient data", socket.EndSend(&ctx),
                static_cast<SIZE_T>(sizeof(UINT_PTR)));
        }

        AssertNoException("socket.Shutdown", socket.Shutdown());
        AssertNoException("socket.Close", socket.Close());

        AssertNoException("Socket::Cleanup", Socket::Cleanup());

        return static_cast<DWORD>(cnt);
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

        LOCK_COUT;
        std::cout << "Client (sender) " 
            << socket.GetPeerEndPoint().ToStringA().PeekBuffer() 
            << " connected." << std::endl;
        UNLOCK_COUT;

        for (UINT_PTR i = 0; i < cnt; i++) {
            LOCK_COUT;
            std::cout << "BeginReceive #" << i << std::endl;
            UNLOCK_COUT;

            ctx.Reset();
            socket.BeginReceive(&data, sizeof(data), &ctx);

            LOCK_COUT;
            std::cout << "Wait (send) #" << i << std::endl;
            UNLOCK_COUT;
            ctx.Wait();

            LOCK_COUT;
            std::cout << "EndReceive #" << i << std::endl;
            UNLOCK_COUT;
            AssertEqual("Received sufficient data", socket.EndReceive(&ctx),
                static_cast<SIZE_T>(sizeof(UINT_PTR)));
        }

        AssertNoException("socket.Shutdown", socket.Shutdown());
        AssertNoException("socket.Close", socket.Close());

        AssertNoException("Socket::Cleanup", Socket::Cleanup());

        return static_cast<DWORD>(cnt);
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
    receiver.Start(reinterpret_cast<void *>(cnt));
    sender.Start(reinterpret_cast<void *>(cnt));

    sender.Join();
    receiver.Join();
}
