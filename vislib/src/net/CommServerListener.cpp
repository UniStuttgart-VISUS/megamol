/*
 * CommServerListener.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/net/CommServerListener.h"
#include "vislib/net/Socket.h"

#include "vislib/Exception.h"
#include "vislib/net/CommServer.h"


/*
 * vislib::net::CommServerListener::~CommServerListener
 */
vislib::net::CommServerListener::~CommServerListener(void) {}


/*
 * vislib::net::CommServerListener::OnServerError
 */
bool vislib::net::CommServerListener::OnServerError(const CommServer& src, const vislib::Exception& exception) throw() {
    return true;
}


/*
 * vislib::net::CommServerListener::OnServerExited
 */
void vislib::net::CommServerListener::OnServerExited(const CommServer& src) throw() {}


/*
 * vislib::net::CommServerListener::OnServerStarted
 */
void vislib::net::CommServerListener::OnServerStarted(const CommServer& src) throw() {}


/*
 * vislib::net::CommServerListener::CommServerListener
 */
vislib::net::CommServerListener::CommServerListener(void) {}
