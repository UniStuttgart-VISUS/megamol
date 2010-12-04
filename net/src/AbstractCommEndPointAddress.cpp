/*
 * AbstractCommEndPointAddress.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractCommEndPointAddress.h"

#include "vislib/StackTrace.h"


/*
 * vislib::net::AbstractCommEndPointAddress::AbstractCommEndPointAddress
 */
vislib::net::AbstractCommEndPointAddress::AbstractCommEndPointAddress(void) 
        : Super() {
    VLSTACKTRACE("AbstractCommEndPointAddress::AbstractCommEndPointAddress",
        __FILE__, __LINE__);
}


/* 
 * vislib::net::AbstractCommEndPointAddress::AbstractCommEndPointAddress
 */
vislib::net::AbstractCommEndPointAddress::AbstractCommEndPointAddress(
        const AbstractCommEndPointAddress& rhs) : Super(rhs) {
    VLSTACKTRACE("AbstractCommEndPointAddress::AbstractCommEndPointAddress",
        __FILE__, __LINE__);
}


/*
 * vislib::net::AbstractCommEndPointAddress::~AbstractCommEndPointAddress
 */
vislib::net::AbstractCommEndPointAddress::~AbstractCommEndPointAddress(void) {
    VLSTACKTRACE("AbstractCommEndPointAddress::~AbstractCommEndPointAddress",
        __FILE__, __LINE__);
}
