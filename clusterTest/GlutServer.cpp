/*
 * GlutServer.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "GlutServer.h"

using namespace vislib::net::cluster;


/*
 * GlutServer::~GlutServer
 */
GlutServer::~GlutServer(void) {
}


/*
 * GlutServer::onFrameRender
 */
//void GlutServer::onFrameRender(void) {
//    ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//}


/*
 * GlutServer::GlutServer
 */
GlutServer::GlutServer(void) : GlutServerNode<GlutServer>() {
}
