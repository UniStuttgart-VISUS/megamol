/*
 * AbstractCommServerChannel.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractCommServerChannel.h"

#include "the/stack_trace.h"


/*
 * vislib::net::AbstractCommServerChannel::AbstractCommServerChannel
 */
vislib::net::AbstractCommServerChannel::AbstractCommServerChannel(void) 
        : Super() {
    THE_STACK_TRACE;
}


/*
 * vislib::net::AbstractCommServerChannel::~AbstractCommServerChannel
 */
vislib::net::AbstractCommServerChannel::~AbstractCommServerChannel(void) {
    THE_STACK_TRACE;
    // Note: Calling Close() here to ensure correct cleanup of all child classes
    // will not work. The child class that has the actual implementation of
    // Close() must do that.
}
