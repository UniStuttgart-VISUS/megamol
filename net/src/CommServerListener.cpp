/*
 * CommServerListener.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/Socket.h"  // Must be first.
#include "vislib/CommServerListener.h"

#include "vislib/CommServer.h"
#include "the/exception.h"
#include "the/stack_trace.h"


/*
 * vislib::net::CommServerListener::~CommServerListener
 */
vislib::net::CommServerListener::~CommServerListener(void) {
    THE_STACK_TRACE;
}


/*
 * vislib::net::CommServerListener::OnServerError
 */
bool vislib::net::CommServerListener::OnServerError(
        const CommServer& src, const the::exception& exception) throw() {
    THE_STACK_TRACE;
    return true;
}


/*
 * vislib::net::CommServerListener::OnServerExited
 */
void vislib::net::CommServerListener::OnServerExited(
        const CommServer& src) throw() {
    THE_STACK_TRACE;

}


/*
 * vislib::net::CommServerListener::OnServerStarted
 */
void vislib::net::CommServerListener::OnServerStarted(
        const CommServer& src) throw() {
    THE_STACK_TRACE;
}


/*
 * vislib::net::CommServerListener::CommServerListener
 */
vislib::net::CommServerListener::CommServerListener(void) {
    THE_STACK_TRACE;
}
