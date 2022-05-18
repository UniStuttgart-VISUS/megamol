/*
 * AbstractCommChannel.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/net/AbstractCommChannel.h"
#include "vislib/net/Socket.h"


/*
 * vislib::net::AbstractCommChannel::TIMEOUT_INFINITE
 */
const UINT vislib::net::AbstractCommChannel::TIMEOUT_INFINITE =
    vislib::net::AbstractCommClientChannel::TIMEOUT_INFINITE;


/*
 * vislib::net::AbstractCommChannel::AbstractCommChannel
 */
vislib::net::AbstractCommChannel::AbstractCommChannel(void)
        : AbstractCommClientChannel()
        , AbstractCommServerChannel() {}


/*
 * vislib::net::AbstractCommChannel::~AbstractCommChannel
 */
vislib::net::AbstractCommChannel::~AbstractCommChannel(void) {
    // Note: Calling Close() here to ensure correct cleanup of all child classes
    // will not work. The child class that has the actual implementation of
    // Close() must do that.
}
