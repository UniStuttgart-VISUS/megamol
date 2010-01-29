/*
 * PeerDisconnectedException.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/PeerDisconnectedException.h"



/*
 * vislib::net::PeerDisconnectedException::FormatMessageForLocalEndpoint
 */
vislib::StringA 
vislib::net::PeerDisconnectedException::FormatMessageForLocalEndpoint(
        const char *localEndPoint) {
    StringA retval;
    retval.Format("The peer end point of \"%s\" disconnected gracefully.",
        localEndPoint);
    return retval;
}


/*
 * vislib::net::PeerDisconnectedException::FormatMessageForLocalEndpoint
 */
vislib::StringW 
vislib::net::PeerDisconnectedException::FormatMessageForLocalEndpoint(
        const wchar_t *localEndPoint) {
    StringW retval;
    retval.Format(L"The peer end point of \"%hs\" disconnected gracefully.",
        localEndPoint);
    return retval;
}


/*
 * vislib::net::PeerDisconnectedException::PeerDisconnectedException
 */
vislib::net::PeerDisconnectedException::PeerDisconnectedException(
        const char *msg, const char *file, const int line) 
        : Super (msg, file, line) {
}


/*
 * vislib::net::PeerDisconnectedException::PeerDisconnectedException
 */
vislib::net::PeerDisconnectedException::PeerDisconnectedException(
        const wchar_t *msg, const char *file, const int line) 
        : Super (msg, file, line) {
}


/*
 * vislib::net::PeerDisconnectedException::PeerDisconnectedException
 */
vislib::net::PeerDisconnectedException::PeerDisconnectedException(
        const PeerDisconnectedException& rhs) : Super (rhs) {
}


/*
 * vislib::net::PeerDisconnectedException::~PeerDisconnectedException
 */
vislib::net::PeerDisconnectedException::~PeerDisconnectedException(void) {
}
