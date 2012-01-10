/*
 * SimpleMessageDispatchListener.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/Socket.h"                          // Must be first!
#include "vislib/SimpleMessageDispatchListener.h"

#include "vislib/AbstractSimpleMessage.h"
#include "vislib/Exception.h"
#include "vislib/SimpleMessageDispatcher.h"
#include "vislib/StackTrace.h"


/*
 * vislib::net::SimpleMessageDispatchListener::SimpleMessageDispatchListener
 */
vislib::net::SimpleMessageDispatchListener::SimpleMessageDispatchListener(
        void) {
    VLSTACKTRACE("SimpleMessageDispatchListener::SimpleMessageDispatchListener",
        __FILE__, __LINE__);
}


/*
 * vislib::net::SimpleMessageDispatchListener::~SimpleMessageDispatchListener
 */
vislib::net::SimpleMessageDispatchListener::~SimpleMessageDispatchListener(
        void) {
    VLSTACKTRACE("SimpleMessageDispatchListener::"
        "~SimpleMessageDispatchListener", __FILE__, __LINE__);
}


/* 
 * vislib::net::SimpleMessageDispatchListener::OnCommunicationError
 */
bool vislib::net::SimpleMessageDispatchListener::OnCommunicationError(
        SimpleMessageDispatcher& src, 
        const vislib::Exception& exception) throw() {
    VLSTACKTRACE("SimpleMessageDispatchListener::OnCommunicationError", 
        __FILE__, __LINE__);
    return true;
}


/*
 * vislib::net::SimpleMessageDispatchListener::OnDispatcherExited
 */
void vislib::net::SimpleMessageDispatchListener::OnDispatcherExited(
        SimpleMessageDispatcher& src) throw() {
    VLSTACKTRACE("SimpleMessageDispatchListener::OnDispatcherExited", 
        __FILE__, __LINE__);
}


/*
 * vislib::net::SimpleMessageDispatchListener::OnDispatcherStarted
 */
void vislib::net::SimpleMessageDispatchListener::OnDispatcherStarted(
        SimpleMessageDispatcher& src) throw() {
    VLSTACKTRACE("SimpleMessageDispatchListener::OnDispatcherStarted", 
        __FILE__, __LINE__);
}


///*
// * vislib::net::SimpleMessageDispatchListener::OnMessageBodyReceived
// */
//bool vislib::net::SimpleMessageDispatchListener::OnMessageBodyReceived(
//        const AbstractSimpleMessageHeader& header, const void *body) throw() {
//    VLSTACKTRACE("SimpleMessageDispatchListener::OnMessageBodyReceived", 
//        __FILE__, __LINE__);
//    return true;
//}
//
//
///*
// * vislib::net::SimpleMessageDispatchListener::OnMessageHeaderReceived
// */
//void vislib::net::SimpleMessageDispatchListener::OnMessageHeaderReceived(
//        void *& outDst, 
//        SimpleMessageSize& outDstSize, 
//        SimpleMessageSize& outOffset, 
//        const AbstractSimpleMessageHeader& header) throw() {
//    VLSTACKTRACE("SimpleMessageDispatchListener::OnMessageHeaderReceived", 
//        __FILE__, __LINE__);    
//    outDst = NULL;
//    outDstSize = 0;
//}