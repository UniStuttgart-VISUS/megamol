/*
 * SimpleMessageDispatchListener.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/SimpleMessageDispatchListener.h"

#include "vislib/AbstractInboundCommChannel.h"
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
        const SimpleMessageDispatcher& src, 
        const vislib::Exception& exception) throw() {
    VLSTACKTRACE("SimpleMessageDispatchListener::OnCommunicationError", 
        __FILE__, __LINE__);
    return true;
}


/*
 * vislib::net::SimpleMessageDispatchListener::OnDispatcherExited
 */
void vislib::net::SimpleMessageDispatchListener::OnDispatcherExited(
        const SimpleMessageDispatcher& src) throw() {
    VLSTACKTRACE("SimpleMessageDispatchListener::OnDispatcherExited", 
        __FILE__, __LINE__);
}


/*
 * vislib::net::SimpleMessageDispatchListener::OnDispatcherStarted
 */
void vislib::net::SimpleMessageDispatchListener::OnDispatcherStarted(
        const SimpleMessageDispatcher& src) throw() {
    VLSTACKTRACE("SimpleMessageDispatchListener::OnDispatcherStarted", 
        __FILE__, __LINE__);
}
