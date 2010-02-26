/*
 * AbstractClientEndPoint.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractClientEndPoint.h"

#include "vislib/StackTrace.h"
#include "vislib/StringConverter.h"


/*
 * vislib::net::AbstractClientEndPoint::Connect
 */
void vislib::net::AbstractClientEndPoint::Connect(const char *address) {
     VLSTACKTRACE("AbstractClientEndPoint::Connect", __FILE__, __LINE__);
     this->Connect(A2W(address));
}


/*
 * vislib::net::AbstractClientEndPoint::AbstractClientEndPoint
 */
vislib::net::AbstractClientEndPoint::AbstractClientEndPoint(void) {
     VLSTACKTRACE("AbstractClientEndPoint::AbstractClientEndPoint", __FILE__,
        __LINE__);
}


/*
 * vislib::net::AbstractClientEndPoint::~AbstractClientEndPoint
 */
vislib::net::AbstractClientEndPoint::~AbstractClientEndPoint(void) {
     VLSTACKTRACE("AbstractClientEndPoint::~AbstractClientEndPoint", __FILE__,
        __LINE__);
}
