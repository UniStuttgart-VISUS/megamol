/*
 * testmsgdisp.cpp
 *
 * Copyright (C) 2006 - 2010 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/TcpCommChannel.h"
#include "testmsgdisp.h"

#include "testhelper.h"
#include "vislib/IPCommEndPoint.h"
#include "vislib/RunnableThread.h"
#include "vislib/Semaphore.h"
#include "vislib/SimpleMessageDispatcher.h"
#include "vislib/SimpleMessageDispatchListener.h"


static const long CNT_MSGS = 1;
static long cntReceived = 0;

vislib::sys::Semaphore semWaitRecv(0l, CNT_MSGS);

class Listener : public vislib::net::SimpleMessageDispatchListener {
    virtual bool OnMessageReceived(vislib::net::SimpleMessageDispatcher& src, const vislib::net::AbstractSimpleMessage& msg) throw();
};

bool Listener::OnMessageReceived(vislib::net::SimpleMessageDispatcher& src, const vislib::net::AbstractSimpleMessage& msg) throw() {
    std::cout << "Received Message " << msg.GetHeader().GetMessageID() << std::endl;
    ++::cntReceived;
    ::semWaitRecv.Unlock();
    return true;
}


void TestTcpDisp(void) {
    using namespace vislib::net;
    const vislib::StringA TEST_ADDRESS("127.0.0.1:12345");

    vislib::sys::RunnableThread<SimpleMessageDispatcher> dispatcher;
    SimpleMessage msg;
    Listener listener;
    vislib::SmartRef<TcpCommChannel> recvChannel = TcpCommChannel::Create();
    vislib::SmartRef<TcpCommChannel> sendChannel(NULL);
    vislib::SmartRef<TcpCommChannel> serverChannel = TcpCommChannel::Create();

    dispatcher.AddListener(&listener);
    ::cntReceived = 0;

    /* Establish the connection. */
    serverChannel->Bind(IPCommEndPoint::Create(TEST_ADDRESS).DynamicCast<AbstractCommEndPoint>());
    serverChannel->Listen();
    recvChannel->Connect(IPCommEndPoint::Create(TEST_ADDRESS).DynamicCast<AbstractCommEndPoint>());
    sendChannel = serverChannel->Accept().DynamicCast<vislib::net::TcpCommChannel>();
    serverChannel->Close();

    /* Start the dispatcher thread. */
    SimpleMessageDispatcher::Configuration config(recvChannel);
    dispatcher.Start(&config);

    msg.GetHeader().SetBodySize(0);
    msg.GetHeader().SetMessageID(27);
    sendChannel->Send(static_cast<void *>(msg), msg.GetMessageSize());

    std::cout << "Waiting for all test messages being received ..." << std::endl;
    for (long i = 0; i < CNT_MSGS; i++) {
        ::semWaitRecv.Lock();
    }

    dispatcher.Terminate(false);

    AssertEqual("Received the expected number of messages using TcpCommChannel", CNT_MSGS, cntReceived);
}


void TestUdpDisp(void) {
    using namespace vislib::net;
    const vislib::StringA TEST_ADDRESS("127.0.0.1:12345");

    vislib::sys::RunnableThread<SimpleMessageDispatcher> dispatcher;
    SimpleMessage msg;
    Listener listener;
    vislib::SmartRef<UdpCommChannel> recvChannel = UdpCommChannel::Create();
    vislib::SmartRef<UdpCommChannel> sendChannel = UdpCommChannel::Create();

    dispatcher.AddListener(&listener);
    ::cntReceived = 0;

    /* "Connect" the end points. */
    recvChannel->Bind(IPCommEndPoint::Create(TEST_ADDRESS).DynamicCast<AbstractCommEndPoint>());
    sendChannel->Connect(IPCommEndPoint::Create(TEST_ADDRESS).DynamicCast<AbstractCommEndPoint>());

    /* Start the dispatcher thread. */
    SimpleMessageDispatcher::Configuration config(recvChannel);
    dispatcher.Start(&config);

    msg.GetHeader().SetBodySize(0);
    msg.GetHeader().SetMessageID(27);
    sendChannel->Send(static_cast<void *>(msg), msg.GetMessageSize());

    std::cout << "Waiting for all test messages being received ..." << std::endl;
    for (long i = 0; i < CNT_MSGS; i++) {
        ::semWaitRecv.Lock();
    }

    dispatcher.Terminate(false);

    AssertEqual("Received the expected number of messages using UdpCommChannel", CNT_MSGS, cntReceived);
}


void TestMsgDisp(void) {
    vislib::net::Socket::Startup();
    ::TestTcpDisp();
    ::TestUdpDisp();
    vislib::net::Socket::Cleanup();
}
