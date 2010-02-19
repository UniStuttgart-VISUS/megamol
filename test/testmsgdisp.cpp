/*
 * testmsgdisp.cpp
 *
 * Copyright (C) 2006 - 2010 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/TcpCommChannel.h"
#include "testmsgdisp.h"

#include "testhelper.h"
#include "vislib/RunnableThread.h"
#include "vislib/Semaphore.h"
#include "vislib/SimpleMessageDispatcher.h"
#include "vislib/SimpleMessageDispatchListener.h"


static const long CNT_MSGS = 1;

vislib::sys::Semaphore semWaitRecv(0l, CNT_MSGS);

class Listener : public vislib::net::SimpleMessageDispatchListener {
    virtual bool OnMessageReceived(const vislib::net::SimpleMessageDispatcher& src, const vislib::net::AbstractSimpleMessage& msg) throw();
};

bool Listener::OnMessageReceived(const vislib::net::SimpleMessageDispatcher& src, const vislib::net::AbstractSimpleMessage& msg) throw() {
    std::cout << "Received Message " << msg.GetHeader().GetMessageID() << std::endl;
    ::semWaitRecv.Unlock();
    return true;
}


void TestMsgDisp(void) {
    const vislib::StringA TEST_ADDRESS("127.0.0.1:12345");

    vislib::net::Socket::Startup();

    vislib::sys::RunnableThread<vislib::net::SimpleMessageDispatcher> dispatcher;
    vislib::net::SimpleMessage msg;
    Listener listener;
    vislib::SmartRef<vislib::net::TcpCommChannel> recvChannel(new vislib::net::TcpCommChannel(), false);
    vislib::SmartRef<vislib::net::TcpCommChannel> sendChannel(NULL);
    vislib::SmartRef<vislib::net::TcpCommChannel> serverChannel(new vislib::net::TcpCommChannel(), false);

    dispatcher.AddListener(&listener);

    /* Establish the connection. */
    serverChannel->Bind(TEST_ADDRESS);
    serverChannel->Listen();
    recvChannel->Connect(TEST_ADDRESS);
    sendChannel = serverChannel->Accept().DynamicCast<vislib::net::TcpCommChannel>();
    serverChannel->Close();

    /* Start the dispatcher thread. */
    vislib::net::AbstractInboundCommChannel *cc = recvChannel.operator ->();
    dispatcher.Start(recvChannel.operator ->());

    msg.GetHeader().SetBodySize(0);
    msg.GetHeader().SetMessageID(27);
    sendChannel->Send(static_cast<void *>(msg), msg.GetMessageSize());

    std::cout << "Waiting for all test messages being received ..." << std::endl;
    for (long i = 0; i < CNT_MSGS; i++) {
        ::semWaitRecv.Lock();
    }

    dispatcher.Terminate(false);

    /* Cleanup network stuff. */
    recvChannel.Release();
    sendChannel.Release();
    serverChannel.Release();
    vislib::net::Socket::Cleanup();
}