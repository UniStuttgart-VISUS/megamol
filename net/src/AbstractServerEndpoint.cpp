/*
 * AbstractServerEndpoint.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractServerEndpoint.h"

#include "vislib/StackTrace.h"
#include "vislib/StringConverter.h"


/*
 * vislib::net::AbstractServerEndpoint::Bind
 */
void vislib::net::AbstractServerEndpoint::Bind(const char *address) {
     VLSTACKTRACE("AbstractServerEndpoint::Bind", __FILE__, __LINE__);
     this->Bind(A2W(address));
}


/*
 * vislib::net::AbstractServerEndpoint::AbstractServerEndpoint
 */
vislib::net::AbstractServerEndpoint::AbstractServerEndpoint(void) {
     VLSTACKTRACE("AbstractServerEndpoint::AbstractServerEndpoint", __FILE__,
        __LINE__);
}


/*
 * vislib::net::AbstractServerEndpoint::~AbstractServerEndpoint
 */
vislib::net::AbstractServerEndpoint::~AbstractServerEndpoint(void) {
     VLSTACKTRACE("AbstractServerEndpoint::~AbstractServerEndpoint", __FILE__,
        __LINE__);
}
