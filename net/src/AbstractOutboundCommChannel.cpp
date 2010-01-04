/*
 * AbstractOutboundCommChannel.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractOutboundCommChannel.h"

#include "vislib/StackTrace.h"


/*
 * vislib::net::AbstractOutboundCommChannel::~AbstractOutboundCommChannel
 */
vislib::net::AbstractOutboundCommChannel::~AbstractOutboundCommChannel(void) {
    VLSTACKTRACE("AbstractOutboundCommChannel::~AbstractOutboundCommChannel",
        __FILE__, __LINE__);
}


/*
 * vislib::net::AbstractOutboundCommChannel::AbstractOutboundCommChannel
 */
vislib::net::AbstractOutboundCommChannel::AbstractOutboundCommChannel(void) {
    VLSTACKTRACE("AbstractOutboundCommChannel::AbstractOutboundCommChannel",
        __FILE__, __LINE__);
}
