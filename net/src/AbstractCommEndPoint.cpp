/*
 * AbstractCommEndPoint.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractCommEndPoint.h"

#include "vislib/StackTrace.h"


/*
 * vislib::net::AbstractCommEndPoint::AbstractCommEndPoint
 */
vislib::net::AbstractCommEndPoint::AbstractCommEndPoint(void) : Super() {
    VLSTACKTRACE("AbstractCommEndPoint::AbstractCommEndPoint",
        __FILE__, __LINE__);
}


/* 
 * vislib::net::AbstractCommEndPoint::AbstractCommEndPoint
 */
vislib::net::AbstractCommEndPoint::AbstractCommEndPoint(
        const AbstractCommEndPoint& rhs) : Super(rhs) {
    VLSTACKTRACE("AbstractCommEndPoint::AbstractCommEndPoint",
        __FILE__, __LINE__);
}


/*
 * vislib::net::AbstractCommEndPoint::~AbstractCommEndPoint
 */
vislib::net::AbstractCommEndPoint::~AbstractCommEndPoint(void) {
    VLSTACKTRACE("AbstractCommEndPoint::~AbstractCommEndPoint",
        __FILE__, __LINE__);
}
