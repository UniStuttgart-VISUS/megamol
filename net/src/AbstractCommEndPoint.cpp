/*
 * AbstractCommEndPoint.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractCommEndPoint.h"

#include "the/stack_trace.h"


/*
 * vislib::net::AbstractCommEndPoint::AbstractCommEndPoint
 */
vislib::net::AbstractCommEndPoint::AbstractCommEndPoint(void) : Super() {
    THE_STACK_TRACE;
}


/* 
 * vislib::net::AbstractCommEndPoint::AbstractCommEndPoint
 */
vislib::net::AbstractCommEndPoint::AbstractCommEndPoint(
        const AbstractCommEndPoint& rhs) : Super(rhs) {
    THE_STACK_TRACE;
}


/*
 * vislib::net::AbstractCommEndPoint::~AbstractCommEndPoint
 */
vislib::net::AbstractCommEndPoint::~AbstractCommEndPoint(void) {
    THE_STACK_TRACE;
}
