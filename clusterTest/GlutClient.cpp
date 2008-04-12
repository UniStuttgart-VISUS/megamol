/*
 * GlutClient.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "GlutClient.h"

using namespace vislib::net::cluster;


/*
 * GlutClient::~GlutClient
 */
GlutClient::~GlutClient(void) {
}


/*
 * GlutClient::GlutClient
 */
GlutClient::GlutClient(void) : GlutClientNode<GlutClient>() {
}
