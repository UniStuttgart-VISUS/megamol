/*
 * AbstractInboundCommChannel.cpp
 *
* Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractInboundCommChannel.h"

#include "vislib/StackTrace.h"


/*
 * vislib::net::AbstractInboundCommChannel::~AbstractInboundCommChannel
 */
vislib::net::AbstractInboundCommChannel::~AbstractInboundCommChannel(void) {
    VLSTACKTRACE("AbstractInboundCommChannel::~AbstractInboundCommChannel",
        __FILE__, __LINE__);
}


/*
 * vislib::net::AbstractInboundCommChannel::AbstractInboundCommChannel
 */
vislib::net::AbstractInboundCommChannel::AbstractInboundCommChannel(void) 
        : Super() {
    VLSTACKTRACE("AbstractInboundCommChannel::AbstractInboundCommChannel",
        __FILE__, __LINE__);
}
