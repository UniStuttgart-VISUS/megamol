/*
 * CommServerListener.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/Socket.h"  // Must be first.
#include "vislib/CommServerListener.h"

#include "vislib/CommServer.h"
#include "vislib/Exception.h"
#include "vislib/StackTrace.h"


/*
 * vislib::net::CommServerListener::~CommServerListener
 */
vislib::net::CommServerListener::~CommServerListener(void) {
    VLSTACKTRACE("CommServerListener::~CommServerListener", __FILE__, __LINE__);
}


/*
 * vislib::net::CommServerListener::OnServerError
 */
bool vislib::net::CommServerListener::OnServerError(
        const CommServer& src, const vislib::Exception& exception) throw() {
    VLSTACKTRACE("CommServerListener::OnServerError", __FILE__, __LINE__);
    return true;
}


/*
 * vislib::net::CommServerListener::OnServerExited
 */
void vislib::net::CommServerListener::OnServerExited(
        const CommServer& src) throw() {
    VLSTACKTRACE("CommServerListener::OnServerExited", __FILE__, __LINE__);

}


/*
 * vislib::net::CommServerListener::OnServerStarted
 */
void vislib::net::CommServerListener::OnServerStarted(
        const CommServer& src) throw() {
    VLSTACKTRACE("CommServerListener::OnServerStarted", __FILE__, __LINE__);
}


/*
 * vislib::net::CommServerListener::CommServerListener
 */
vislib::net::CommServerListener::CommServerListener(void) {
    VLSTACKTRACE("CommServerListener::CommServerListener", __FILE__, __LINE__);
}
