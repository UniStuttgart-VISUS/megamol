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
#include "the/stack_trace.h"


/*
 * vislib::net::SimpleMessageDispatchListener::SimpleMessageDispatchListener
 */
vislib::net::SimpleMessageDispatchListener::SimpleMessageDispatchListener(
        void) {
    THE_STACK_TRACE;
}


/*
 * vislib::net::SimpleMessageDispatchListener::~SimpleMessageDispatchListener
 */
vislib::net::SimpleMessageDispatchListener::~SimpleMessageDispatchListener(
        void) {
    THE_STACK_TRACE;
}


/* 
 * vislib::net::SimpleMessageDispatchListener::OnCommunicationError
 */
bool vislib::net::SimpleMessageDispatchListener::OnCommunicationError(
        SimpleMessageDispatcher& src, 
        const vislib::Exception& exception) throw() {
    THE_STACK_TRACE;
    return true;
}


/*
 * vislib::net::SimpleMessageDispatchListener::OnDispatcherExited
 */
void vislib::net::SimpleMessageDispatchListener::OnDispatcherExited(
        SimpleMessageDispatcher& src) throw() {
    THE_STACK_TRACE;
}


/*
 * vislib::net::SimpleMessageDispatchListener::OnDispatcherStarted
 */
void vislib::net::SimpleMessageDispatchListener::OnDispatcherStarted(
        SimpleMessageDispatcher& src) throw() {
    THE_STACK_TRACE;
}


///*
// * vislib::net::SimpleMessageDispatchListener::OnMessageBodyReceived
// */
//bool vislib::net::SimpleMessageDispatchListener::OnMessageBodyReceived(
//        const AbstractSimpleMessageHeader& header, const void *body) throw() {
//    THE_STACK_TRACE;
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
//    THE_STACK_TRACE;    
//    outDst = NULL;
//    outDstSize = 0;
//}