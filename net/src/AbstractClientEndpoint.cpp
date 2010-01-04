/*
 * AbstractClientEndpoint.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractClientEndpoint.h"

#include "vislib/StackTrace.h"
#include "vislib/StringConverter.h"


/*
 * vislib::net::AbstractClientEndpoint::Connect
 */
void vislib::net::AbstractClientEndpoint::Connect(const char *address) {
     VLSTACKTRACE("AbstractClientEndpoint::Connect", __FILE__, __LINE__);
     this->Connect(A2W(address));
}


/*
 * vislib::net::AbstractClientEndpoint::AbstractClientEndpoint
 */
vislib::net::AbstractClientEndpoint::AbstractClientEndpoint(void) {
     VLSTACKTRACE("AbstractClientEndpoint::AbstractClientEndpoint", __FILE__,
        __LINE__);
}


/*
 * vislib::net::AbstractClientEndpoint::~AbstractClientEndpoint
 */
vislib::net::AbstractClientEndpoint::~AbstractClientEndpoint(void) {
     VLSTACKTRACE("AbstractClientEndpoint::~AbstractClientEndpoint", __FILE__,
        __LINE__);
}
