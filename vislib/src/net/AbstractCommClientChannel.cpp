/*
 * AbstractCommClientChannel.cpp
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/net/AbstractCommClientChannel.h"
#include "vislib/net/Socket.h"


/*
 * vislib::net::AbstractCommClientChannel::TIMEOUT_INFINITE
 */
const UINT vislib::net::AbstractCommClientChannel::TIMEOUT_INFINITE = vislib::net::Socket::TIMEOUT_INFINITE;


/*
 * vislib::net::AbstractCommClientChannel::AbstractCommClientChannel
 */
vislib::net::AbstractCommClientChannel::AbstractCommClientChannel(void) : Super() {}


/*
 * vislib::net::AbstractCommClientChannel::~AbstractCommClientChannel
 */
vislib::net::AbstractCommClientChannel::~AbstractCommClientChannel(void) {
    // Note: Calling Close() here to ensure correct cleanup of all child classes
    // will not work. The child class that has the actual implementation of
    // Close() must do that.
}
