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
#include "vislib/SimpleMessageDispatcher.h"
#include "vislib/SimpleMessageDispatchListener.h"



class Listener : public vislib::net::SimpleMessageDispatchListener {
    virtual bool OnMessageReceived(const vislib::net::SimpleMessageDispatcher& src, const vislib::net::AbstractSimpleMessage& msg) throw();
};

bool Listener::OnMessageReceived(const vislib::net::SimpleMessageDispatcher& src, const vislib::net::AbstractSimpleMessage& msg) throw() {
    std::cout << "Received Message " << msg.GetHeader().GetMessageID() << std::endl;
    return true;
}


void TestMsgDisp(void) {
    vislib::sys::RunnableThread<vislib::net::SimpleMessageDispatcher> dispatcher;
    Listener listener;
    vislib::SmartRef<vislib::net::TcpCommChannel> recvChannel(new vislib::net::TcpCommChannel());
    vislib::SmartRef<vislib::net::TcpCommChannel> sendChannel(NULL);
    vislib::SmartRef<vislib::net::TcpCommChannel> serverChannel(new vislib::net::TcpCommChannel());


    dispatcher.AddListener(&listener);

    serverChannel->Bind("127.0.0.1:12345");
    sendChannel = serverChannel->WaitForClient().DynamicCast<vislib::net::TcpCommChannel>();
    serverChannel->Close();
    serverChannel->Release();
    
    


    


}