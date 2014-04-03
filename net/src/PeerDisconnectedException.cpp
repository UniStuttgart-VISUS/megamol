/*
 * PeerDisconnectedException.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/PeerDisconnectedException.h"
#include "the/string.h"
#include "the/text/string_builder.h"


/*
 * vislib::net::PeerDisconnectedException::FormatMessageForLocalEndpoint
 */
the::astring 
vislib::net::PeerDisconnectedException::FormatMessageForLocalEndpoint(
        const char *localEndPoint) {
    the::astring retval;
    the::text::astring_builder::format_to(retval, "The peer end point of \"%s\" disconnected gracefully.",
        localEndPoint);
    return retval;
}


/*
 * vislib::net::PeerDisconnectedException::FormatMessageForLocalEndpoint
 */
the::wstring 
vislib::net::PeerDisconnectedException::FormatMessageForLocalEndpoint(
        const wchar_t *localEndPoint) {
    the::wstring retval;
    the::text::wstring_builder::format_to(retval, L"The peer end point of \"%hs\" disconnected gracefully.",
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
vislib::net::PeerDisconnectedException::~PeerDisconnectedException(void) throw() {
}
