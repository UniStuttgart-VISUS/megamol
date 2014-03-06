/*
 * AbstractCommClientChannel.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/Socket.h"  // Note: For enforcing Winsock2
#include "vislib/AbstractCommClientChannel.h"

#include "the/stack_trace.h"


/*
 * vislib::net::AbstractCommClientChannel::TIMEOUT_INFINITE
 */
const unsigned int vislib::net::AbstractCommClientChannel::TIMEOUT_INFINITE 
    = vislib::net::Socket::TIMEOUT_INFINITE;


/*
 * vislib::net::AbstractCommClientChannel::AbstractCommClientChannel
 */
vislib::net::AbstractCommClientChannel::AbstractCommClientChannel(void) 
        : Super() {
    THE_STACK_TRACE;
}


/*
 * vislib::net::AbstractCommClientChannel::~AbstractCommClientChannel
 */
vislib::net::AbstractCommClientChannel::~AbstractCommClientChannel(void) {
    THE_STACK_TRACE;
    // Note: Calling Close() here to ensure correct cleanup of all child classes
    // will not work. The child class that has the actual implementation of
    // Close() must do that.
}
