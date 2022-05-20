/*
 * SimpleMessageDispatchListener.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "mmcore/utility/net/SimpleMessageDispatchListener.h"
#include "vislib/net/Socket.h"

#include "mmcore/utility/net/AbstractSimpleMessage.h"
#include "mmcore/utility/net/SimpleMessageDispatcher.h"
#include "vislib/Exception.h"


/*
 * vislib::net::SimpleMessageDispatchListener::SimpleMessageDispatchListener
 */
vislib::net::SimpleMessageDispatchListener::SimpleMessageDispatchListener(void) {}


/*
 * vislib::net::SimpleMessageDispatchListener::~SimpleMessageDispatchListener
 */
vislib::net::SimpleMessageDispatchListener::~SimpleMessageDispatchListener(void) {}


/*
 * vislib::net::SimpleMessageDispatchListener::OnCommunicationError
 */
bool vislib::net::SimpleMessageDispatchListener::OnCommunicationError(
    SimpleMessageDispatcher& src, const vislib::Exception& exception) throw() {
    return true;
}


/*
 * vislib::net::SimpleMessageDispatchListener::OnDispatcherExited
 */
void vislib::net::SimpleMessageDispatchListener::OnDispatcherExited(SimpleMessageDispatcher& src) throw() {}


/*
 * vislib::net::SimpleMessageDispatchListener::OnDispatcherStarted
 */
void vislib::net::SimpleMessageDispatchListener::OnDispatcherStarted(SimpleMessageDispatcher& src) throw() {}


///*
// * vislib::net::SimpleMessageDispatchListener::OnMessageBodyReceived
// */
//bool vislib::net::SimpleMessageDispatchListener::OnMessageBodyReceived(
//        const AbstractSimpleMessageHeader& header, const void *body) throw() {
////    return true;
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
////    outDst = NULL;
//    outDstSize = 0;
//}
