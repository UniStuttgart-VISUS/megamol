/*
 * AbstractServerEndPoint.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractServerEndPoint.h"

#include "vislib/StackTrace.h"
#include "vislib/StringConverter.h"


/*
 * vislib::net::AbstractServerEndPoint::Bind
 */
void vislib::net::AbstractServerEndPoint::Bind(const char *address) {
     VLSTACKTRACE("AbstractServerEndPoint::Bind", __FILE__, __LINE__);
     this->Bind(A2W(address));
}


/*
 * vislib::net::AbstractServerEndPoint::AbstractServerEndPoint
 */
vislib::net::AbstractServerEndPoint::AbstractServerEndPoint(void) {
     VLSTACKTRACE("AbstractServerEndPoint::AbstractServerEndPoint", __FILE__,
        __LINE__);
}


/*
 * vislib::net::AbstractServerEndPoint::~AbstractServerEndPoint
 */
vislib::net::AbstractServerEndPoint::~AbstractServerEndPoint(void) {
     VLSTACKTRACE("AbstractServerEndPoint::~AbstractServerEndPoint", __FILE__,
        __LINE__);
}
